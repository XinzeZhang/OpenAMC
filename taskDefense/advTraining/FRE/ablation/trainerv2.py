from models.nn._baseTrainer import AnnealingTrainer
import torch.nn.functional as F
import torch
from tqdm.auto import trange
from tqdm.auto import tqdm

class ftmlTrainer(AnnealingTrainer):
    '''
    Create antonymous signals from the same SNR but with different labels.
    Random for each training epoch.
    '''
    def collect_snr(self, train_loader):
        """
        Collect SNR from train_loader.
        """
        snr_list = []
        sig_list = []
        lab_list = []

        for data_batch in train_loader:
            # if hasattr(self.hyper, 'using_snr') and self.hyper.using_snr
            # there will be three elements in data_batch, which are sig_batch, lab_batch, and snr_batch
            sig_batch, lab_batch, snr_batch = data_batch[0], data_batch[1], data_batch[2]
            snr_list.append(snr_batch)
            sig_list.append(sig_batch)
            lab_list.append(lab_batch)


        snr_list = torch.cat(snr_list).to(self.hyper.device)
        sig_list = torch.cat(sig_list).to(self.hyper.device)
        lab_list = torch.cat(lab_list).to(self.hyper.device)
        # idx_list = torch.arange(snr_list.size(0)).to(self.hyper.device)

        self.snr_data_dict = {}
        with trange(len(self.hyper.snr_envs), desc='Collecting SNR', mininterval=0.3, colour='blue', leave=False) as bbar:
            for snr in self.hyper.snr_envs:
                idx_i = torch.where(snr_list == snr)
                # print(snr,snr_list[idx_i[0]])
                sig_i = sig_list[idx_i]
                lab_i = lab_list[idx_i]
                self.snr_data_dict[snr] = (sig_i, lab_i, idx_i[0])
                bbar.update(1)

        return sig_list, lab_list, snr_list

    def repack_train_loader(self, train_loader):
        """
        Repack train_loader to include snr_batch.
        """
        self.sig_list, self.lab_list, self.snr_list = self.collect_snr(train_loader)

        data_size = self.sig_list.size(0)

        antomy_list = []
        for i in trange(data_size):
            snr = self.snr_list[i].item()
            current_label = self.lab_list[i]
            same_snr_data = self.snr_data_dict[snr]
            (sig_snr_data, lab_snr_data, idx_snr_data) = same_snr_data

            # Find indices where labels are different from current label
            different_label_mask = torch.ne(lab_snr_data, current_label)
            different_label_indices = torch.where(different_label_mask)[0]

            set_index = idx_snr_data[different_label_indices]

            antomy_list.append(set_index)

        self.antomy_list = torch.stack(antomy_list).to(self.hyper.device)

        # Repack train_loader to include snr_batch.
        train_set = (self.sig_list, self.lab_list, self.snr_list, self.antomy_list)
        from torch.utils.data import TensorDataset, DataLoader
        train_set = TensorDataset(*train_set)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.hyper.batch_size,
            shuffle=True,
            num_workers=0
        )

        return train_loader

    def loop(self, model,
                 train_loader,
                 val_loader,):

        self.logger.info(f'Using Trainer: {self.__class__.__name__}')
        self.val_loader = val_loader
        self.model = model.to(self.hyper.device)
        self.before_train()

        if self.enable_ftml:
            self.train_loader = self.repack_train_loader(train_loader) # each batch should be a tuple of (sig_batch, lab_batch, snr_batch), todo: check if repack is necessary, due the default pack_data_loader has pack the snr_batch
        else:
            self.train_loader = train_loader # each batch should be a tuple of (sig_batch, lab_batch, snr_batch) --- IGNORE ---

        for self.iter in trange(1, self.hyper.epochs + 1):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

        # last_model_name = self.hyper.data_name + '_' + \
        #     f'{self.hyper.model_name}' + '.early_stop.pt'
        self.early_stopping.early_stop = True
        torch.save(self.early_stopping.early_stop, self.finishTag_file)

    def before_train(self):
        super().before_train()
        self.ftml_beta = self.hyper.ftml_beta if 'ftml_beta' in vars(self.hyper) else 0.05
        self.k_s = self.hyper.k_s if 'k_s' in vars(self.hyper) else 5
        self.k_n = self.hyper.k_n if 'k_n' in vars(self.hyper) else 5
        self.ftml_alpha = self.hyper.ftml_alpha if 'ftml_alpha' in vars(self.hyper) else 1.0
        self.dis_m = self.hyper.dis_m if 'dis_m' in vars(self.hyper) else 'MSE'
        self.enable_ftml = self.hyper.ftml if 'ftml' in vars(self.hyper) else True
        self.mul_acc = True


    def run_optim_step(self, i, data_batch):
        # sig_batch, lab_batch = data_batch[0], data_batch[1]
        loss, acc = self.cal_loss_acc(data_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.model.check_weights_unchanged()

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)

    def cal_tpl_loss(self, data_batch):

        sig_batch, lab_batch, snr_batch = data_batch[0], data_batch[1], data_batch[2]
        antony_batch = data_batch[3]

        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)
        snr_batch = snr_batch.to(self.hyper.device)
        antony_batch = antony_batch.to(self.hyper.device)
        # idx_batch = idx_batch.to(self.hyper.device)

        # ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        batch_size = sig_batch.size(0)
        sig_length = sig_batch.size(2)
        k_s = self.k_s
        k_n = self.k_n
        alpha = self.ftml_alpha
        embedding = self.model.feature_extract #使用feature_extract层嵌入

        def create_synonyms():
            # Generate random shifts for all batch samples and k_s synonyms at once
            shifts = torch.randint(0, sig_length, (batch_size, k_s), device=sig_batch.device)
            # Expand sig_batch to (batch_size, k_s, channels, sig_length)
            expanded_sig = sig_batch.unsqueeze(1).expand(-1, k_s, -1, -1)

            # Create shifted indices for vectorized rolling; shifts[i,j] is the shift amount for expanded_sig[i,j,:,:]
            time_indices = torch.arange(sig_length, device=sig_batch.device)
            # Broadcast shifts to match the signal dimensions
            shifts_expanded = shifts.unsqueeze(-1).unsqueeze(-1)  # (batch_size, k_s, 1, 1)
            # Create rolled indices: (original_idx - shift) % sig_length
            rolled_indices = (time_indices - shifts_expanded) % sig_length  # (batch_size, k_s, 1, sig_length)
            rolled_indices = rolled_indices.expand(-1, -1, sig_batch.size(1), -1)  # (batch_size, k_s, channels, sig_length)
            # Use gather to apply all shifts at once
            synonyms = expanded_sig.gather(-1, rolled_indices)

            return synonyms

        def create_antonymous_v2():
            """
            Create antonymous signals from the same SNR but with different labels.
            Random for each training epoch.
            Prioritizes signals with different labels from the current signal.
            """
            # Faster vectorized sampling of k_n elements from antony_batch along dim=1
            batch_size_antony, seq_len = antony_batch.shape
            # Generate random indices for the first k_n positions only
            rand_indices = torch.argsort(torch.rand(batch_size_antony, seq_len), dim=1)[:, :k_n].to(antony_batch.device)
            # Directly sample k_n elements for all batch samples at once
            sampled_antony_batch = antony_batch.gather(1, rand_indices)  # Shape: (batch_size, k_n)

            flat_indices = sampled_antony_batch.flatten()  # Shape: (batch_size * k_n,)
            flat_antony_sigs = self.sig_list[flat_indices]  # Shape: (batch_size * k_n, signal_dims...)
            # Reshape back to (batch_size, k_n, signal_dims...)
            signal_shape = flat_antony_sigs.shape[1:]  # Get signal dimensions
            antonymous = flat_antony_sigs.view(batch_size, k_n, *signal_shape)

            masks = torch.ones(batch_size, k_n, device=self.hyper.device).float()
            # for i_b in range(batch_size):
            #     # antony_batch_indices = sampled_antony_batch[i_b]
            #     indices = sampled_antony_batch[i_b]
            #     antony_sigs_ib = self.sig_list[indices]
            #     mask = torch.ones(k_n, device=self.hyper.device).float()
            #     antonymous.append(antony_sigs_ib)
            #     masks.append(mask)
            # antonymous = torch.stack(antonymous).to(self.hyper.device)
            # masks = torch.stack(masks).to(self.hyper.device)

            return antonymous, masks
                # if the antonymous signals are not equal to the same label of current signal, the mask is one.
                # actually, the mask is used as the weights for antonymous signals,

                ## Checking code for antonymous signals
                # selected_labels = self.lab_list[indices]
                # selected_snrs = self.snr_list[indices]
                # current_label = lab_batch[i_b]
                # snr = snr_batch[i_b]
                # current_label_in_selected = torch.any(selected_labels == current_label)
                # all_snrs_same = torch.all(selected_snrs == snr)
                # # print(f"Current label: {current_label.item()}")
                # print(f"Current label in selected: {current_label_in_selected.item()}")
                # if current_label_in_selected:
                #     raise ValueError(
                #         f"Current label {current_label.item()} is in the selected antonymous samples. "
                #         "This should not happen, as antonymous samples should have different labels."
                #     )
                # # print(f"Current SNR: {snr}")
                # print(f"All selected SNRs same as current: {all_snrs_same.item()}")
                # if not all_snrs_same:
                #     raise ValueError(
                #         f"Not all selected SNRs are the same as current SNR {snr}. "
                #         "This should not happen, as antonymous samples should have the same SNR."
                #     )


        anchor = embedding(sig_batch).reshape(batch_size, -1)

        synonyms = create_synonyms()  # 近义词
        positive = embedding(synonyms.reshape(-1, synonyms.shape[-2], synonyms.shape[-1]))
        positive = positive.reshape(batch_size, k_s, -1)

        antonymous, antonymous_mask = create_antonymous_v2()  # non_synonym_mask用来去除同标签
        negtive = embedding(antonymous.reshape(-1, antonymous.shape[-2], antonymous.shape[-1]))
        negtive = negtive.reshape(batch_size, k_n, -1)

        if self.dis_m == 'MSE':
            syn_dis = torch.linalg.norm(anchor.unsqueeze(1) - positive, dim=-1, ord=2)
            nonsyn_dis = torch.linalg.norm(anchor.unsqueeze(1) - negtive, dim=-1, ord=2) * antonymous_mask

        elif self.dis_m == 'cos':
            syn_dis = F.cosine_similarity(anchor.unsqueeze(1), positive, dim=-1)
            nonsyn_dis = F.cosine_similarity(anchor.unsqueeze(1), negtive, dim=-1) * antonymous_mask

        else:
            raise Exception('Unsupported distance mertic: {}'.format(self.dis_m))

        tpl_loss = torch.mean(syn_dis, dim=-1) - torch.mean(
            torch.min(torch.zeros_like(nonsyn_dis) + alpha, nonsyn_dis), dim=-1) + alpha
        tpl_loss = torch.mean(F.relu(tpl_loss))

        return tpl_loss

    def cal_loss_acc(self, data_batch):
        sig_batch, lab_batch = data_batch[0], data_batch[1]
        # snr_batch, idx_batch = data_batch[2], data_batch[3]

        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        # ori_loss = 0
        tpl_loss = 0 if not self.enable_ftml else self.cal_tpl_loss(data_batch) #ensure the tpl loss does not affect the awn model.
        loss = ori_loss + self.ftml_beta * tpl_loss
        acc = self.cal_acc(sig_batch, lab_batch)

        return loss, acc

    def cal_ori_loss_acc(self, data_batch):
        sig_batch, lab_batch = data_batch[0], data_batch[1]
        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)

        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        acc = self.cal_acc(sig_batch, lab_batch)
        return ori_loss, acc

    def run_val_step(self,):
        '''
        run validation step, return average loss and accuracy at this epoch
        '''

        with tqdm(total=len(self.val_loader),
                desc=f'Epoch {self.iter}/{self.hyper.epochs}',
                postfix=dict,
                mininterval=0.3,
                colour='blue') as pbar:
            for step, data_batch in enumerate(self.val_loader):
                with torch.no_grad():
                    loss, acc = self.cal_ori_loss_acc(data_batch)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

        return self.val_loss.avg, self.val_acc.avg