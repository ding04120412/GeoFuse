import torch
import torch.nn as nn
import numpy as np
import geoopt, os, copy, time
from torch_geometric.utils import negative_sampling

from config import args
from utils.data_utils import loader, prepare_train_test_data
from utils.curvature_utils import precompute_deltas 
import utils.util as util
from model.geofuse import GeoFuse
from loss import MEL 

util.init_logger(args.log_file)
logger = util.get_logger()
util.set_random(args.seed)

class Trainer(object):
    def __init__(self):
        
        self.overall_start_time = time.perf_counter() 

        self.data = loader(args)
        if self.data is None: exit()
        args.num_nodes = self.data['num_nodes']
        
        mode_label = "SPARSE" if args.num_nodes > 2000 else "DENSE"
        logger.info(f"--- [STATUS] Using {mode_label} Mode for {args.dataset} ---")

        self.use_embedding = self.data.get('weights') is None
        if self.use_embedding:
            args.nfeat = args.nhid
            self.node_emb = nn.Embedding(args.num_nodes, args.nhid).to(args.device)
            nn.init.xavier_uniform_(self.node_emb.weight)
        else:
            args.nfeat = self.data['weights'][0].shape[1]

        delta_path = os.path.join(args.output_path, f'{args.dataset}_deltas.pt')
        if os.path.exists(delta_path):
            self.deltas = torch.load(delta_path, map_location=args.device)
        else:
            self.deltas = [d.to(args.device) for d in precompute_deltas(self.data)]
            torch.save(self.deltas, delta_path)

        self.train_shots = list(range(0, self.data['time_length'] - args.test_length))
        self.test_shots = list(range(self.data['time_length'] - args.test_length, self.data['time_length']))
        self.model = GeoFuse(args).to(args.device)
        self.loss_fn = MEL(args, self.model.c).to(args.device)
        
        
        self._precompute_all_ppr()
        
        params = list(set(list(self.model.parameters()) + list(self.loss_fn.parameters()) + 
                     (list(self.node_emb.parameters()) if self.use_embedding else [])))
        self.optimizer = geoopt.optim.radam.RiemannianAdam(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        

    def _precompute_all_ppr(self):
        logger.info("--- [STATUS] Checking PPR Offline Pre-computation ---")
        all_edges = []
        for t in range(self.data['time_length']):
            edge_index, _, _, _, _ = prepare_train_test_data(self.data, t, args.device)
            all_edges.append(edge_index)
        if not all_edges: return
        for module in self.model.modules():
            if module.__class__.__name__ == 'GAR':
                module.build_offline_ppr(all_edges, args.num_nodes)
        torch.cuda.empty_cache()
        logger.info("--- [STATUS] PPR Offline Pre-computation Completed! ---")

    def train(self):
        logger.info(f"--- [START] Training (Temporal: t->t+1) on {args.dataset} ---")
        best_score, best_res = 0, [0, 0, 0, 0]
        patience_cnt, best_model_wts = 0, copy.deepcopy(self.model.state_dict())
        
        for epoch in range(1, args.max_epoch + 1):
            self.model.train(); self.model.init_history()
            epoch_losses = []
            
            for i in range(len(self.train_shots) - 1):
                t, t_next = self.train_shots[i], self.train_shots[i+1]
                edge_index_t, pos_idx_t, _, _, _ = prepare_train_test_data(self.data, t, args.device)
                x = self.node_emb.weight if self.use_embedding else self.data['weights'][t].to(args.device)
                _, pos_idx_tp1, _, _, _ = prepare_train_test_data(self.data, t_next, args.device)

                self.optimizer.zero_grad()
                z_hyp, _ = self.model(x, edge_index_t, self.deltas[t])
                
                
                loss = self.loss_fn(z_hyp, target_idx=pos_idx_tp1, ref_idx=pos_idx_t, epoch=epoch)
                
               
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                self.optimizer.step()
                epoch_losses.append(loss.item())
            
            self.scheduler.step()
            
            if epoch % args.log_interval == 0:
                te_res = self.evaluate()
                mean_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else 0.0
                logger.info(f'Ep:{epoch:03d} | Loss:{mean_loss:.4f}')
                logger.info(f'   Test AUC:{te_res[0]:.4f} AP:{te_res[1]:.4f} NewAUC:{te_res[2]:.4f} NewAP:{te_res[3]:.4f}')

                current_score = np.mean(te_res)
                if current_score > best_score:
                    best_score, best_res = current_score, te_res 
                    patience_cnt = 0
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                else:
                    patience_cnt += 1
                if patience_cnt >= args.patience: break

        self.model.load_state_dict(best_model_wts)
        
       
        total_elapsed = time.perf_counter() - self.overall_start_time
        
        logger.info("=" * 75)
        logger.info(f"FINAL BEST RESULTS ON {args.dataset.upper()}")
        logger.info("-" * 75)
        logger.info(f"Best AUC:      {best_res[0]:.4f}")
        logger.info(f"Best AP:       {best_res[1]:.4f}")
        logger.info(f"Best NewAUC:   {best_res[2]:.4f}")
        logger.info(f"Best NewAP:    {best_res[3]:.4f}")
        logger.info(f"Total Time:    {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f} min)")
        logger.info("=" * 75)

    def evaluate(self):
        self.model.eval()
        res_list = []
        eval_input_shots = [self.train_shots[-1]] + self.test_shots[:-1]
        eval_target_shots = self.test_shots
        
        with torch.no_grad():
            self.model.init_history()
            for t in self.train_shots[:-1]:
                edge_index, _, _, _, _ = prepare_train_test_data(self.data, t, args.device)
                x = self.node_emb.weight if self.use_embedding else self.data['weights'][t].to(args.device)
                self.model(x, edge_index, self.deltas[t])

            for i in range(len(eval_input_shots)):
                t_in, t_tar = eval_input_shots[i], eval_target_shots[i]
                edge_index_in, _, _, _, _ = prepare_train_test_data(self.data, t_in, args.device)
                x_in = self.node_emb.weight if self.use_embedding else self.data['weights'][t_in].to(args.device)
                _, p_idx_tar, n_idx_tar, np_idx_tar, nn_idx_tar = prepare_train_test_data(self.data, t_tar, args.device)
                
                z, _ = self.model(x_in, edge_index_in, self.deltas[t_in])
                if n_idx_tar is None: n_idx_tar = negative_sampling(p_idx_tar, num_nodes=args.num_nodes)
                a, ap = self.loss_fn.predict(z, p_idx_tar, n_idx_tar)
                if np_idx_tar is not None and np_idx_tar.size(1) > 0:
                    if nn_idx_tar is None: nn_idx_tar = negative_sampling(np_idx_tar, num_nodes=args.num_nodes)
                    na, nap = self.loss_fn.predict(z, np_idx_tar, nn_idx_tar)
                else: na, nap = a, ap
                res_list.append([a, ap, na, nap])
        return np.mean(res_list, axis=0)

if __name__ == '__main__':
    Trainer().train()