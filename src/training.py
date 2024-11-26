import pandas as pd

import os
import torch


def run_seal(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_seal(local_epoch, args.rho)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

        # write to log files
        if c_round % 5 == 0:
            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()
                summary_writer.add_scalar('Test/Acc/user' + str(idx + 1), acc, c_round)
                summary_writer.add_scalar('Test/Loss/user' + str(idx + 1), loss, c_round)

        # save model
        if c_round % 200 == 0:
            if args.data_group == None:
                if args.fgd:
                    for client in clients:
                        torch.save({
                            'c_round': c_round,
                            'model_state_dict': client.model.state_dict(),
                            'optimizer_state_dict': client.optimizer.state_dict(),
                            'trainingLoss': client.train_stats['trainingLosses'],
                        }, os.path.join(args.modelpath, args.OneDataset, args.alg, f'{args.repeat}_localmodel{client.id}_fgd.pt'))
                else:
                    for client in clients:
                        torch.save({
                            'c_round': c_round,
                            'model_state_dict': client.model.state_dict(),
                            'optimizer_state_dict': client.optimizer.state_dict(),
                            'trainingLoss': client.train_stats['trainingLosses'],
                        }, os.path.join(args.modelpath, args.OneDataset, args.alg, f'{args.repeat}_localmodel{client.id}.pt'))
            if args.OneDataset == None:
                if args.fgd:
                    for client in clients:
                        torch.save({
                            'c_round': c_round,
                            'model_state_dict': client.model.state_dict(),
                            'optimizer_state_dict': client.optimizer.state_dict(),
                            'trainingLoss': client.train_stats['trainingLosses'],
                        }, os.path.join(args.modelpath, args.data_group, args.alg, f'{args.repeat}_localmodel{client.id}_fgd.pt'))
                else:
                    for client in clients:
                        torch.save({
                            'c_round': c_round,
                            'model_state_dict': client.model.state_dict(),
                            'optimizer_state_dict': client.optimizer.state_dict(),
                            'trainingLoss': client.train_stats['trainingLosses'],
                        }, os.path.join(args.modelpath, args.data_group, args.alg, f'{args.repeat}_localmodel{client.id}.pt'))
                
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame