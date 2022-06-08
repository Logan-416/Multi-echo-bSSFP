%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GRAPPA-based reconstruction method for multi-echo bSSFP (ETL = 3) data used in our MRM paper (DOI: 10.1002/mrm.29301)
% We thank Dr. Berkin Bilgic for sharing the codes of GRAPPA at https://www.martinos.org/?berkin/software.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
scl % Set [path to] Current Location
cd ..
addpath(genpath('.'))


%% Load multi-echo bSSFP and single-echo bSSFP (reference) data

ETL = 3;

ref_filepath = [pwd,'\sample_data\ETL1\'];
ref_sequence = char('ETL1_1x1_sl3_FA30_meas20_BW1002_LR_FID1698.mat');

filepath = [pwd,'\sample_data\ETL',num2str(ETL),'\'];
all_sequence_dirs = dir(filepath);

for seq = 3:size(all_sequence_dirs,1)               
    name = strtrim(all_sequence_dirs(seq).name);
    multi_struct = load ([filepath,name]);
    multi_name = fieldnames(multi_struct);
    multi_RawData = getfield(multi_struct,char(multi_name));
    size(multi_RawData); % 416   210    32    20     3
    kspace_data_all = repmat(multi_RawData,[1 1 1 1 1 ETL]); % ETL = 3
    size(kspace_data_all); % 416   210    32    20     3     3
  
    single_ref_meas = 20;
    single_struct = load ([ref_filepath, ref_sequence]);
    single_name = fieldnames(single_struct);
    single_RawDate_all = getfield(single_struct,char(single_name));
    size(single_RawDate_all); % 416   208    32    20     3
    single_RawDate_selected = squeeze(single_RawDate_all(:,:,:,single_ref_meas,:));
    size(single_RawDate_selected); % 416   208    32     3
    

%% Apply GRAPPA reconstruction
    
    mkdir([pwd,'\recon_nii\',name(1:end-4)]);
    for meas = 1:size(multi_RawData,4)
        disp(['                        ======',num2str(meas),'/',num2str(size(multi_RawData,4)),'======']);
        for sl = 1:size(multi_RawData,5)
            kspace_data = squeeze(kspace_data_all(:,:,:,meas,sl,:));
            size(kspace_data); % 416   210    32     3
            [N(1), N(2), num_chan, num_cycle] = size(kspace_data);
            
            single_RawDate = squeeze(single_RawDate_selected(:,:,:,sl));
            size(single_RawDate); % 416   208    32
            single_acs = single_RawDate;
            
            %%-------------------------------------------------------------------------
            % Grappa with 1D acceleration
            %--------------------------------------------------------------------------
            
            Ry = num_cycle;                % acceleration factor           
            num_acsX = size(single_acs,1); % acs size in readout
            num_acsY = 64;                 % acs size in phase encoding          
            lambda_tik = 1e-3;             % Tikhonov reg parameter for GRAPPA kernel estimation          
            
            % sampling and acs masks
            mask = zeros([N,num_cycle]);
            del_step = 1;
            del = mod((0:num_cycle-1) * del_step, Ry);
            for t = 1:num_cycle
                mask(:,1+del(t):Ry:end,t) = 1;
            end
            
            kernel_size = [5,3];       % assume odd kernel size [kx,ky]           
            kernel_hsize = (kernel_size-1)/2;     
            
            pad_size = kernel_hsize .* [1,Ry];
            N_pad = N + 2*pad_size;
            
            % k-space limits for training:
            ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region          
            ky_end = num_acsY - Ry * kernel_hsize(2);  % last kernel center point that fits acs region           
            kx_begin = 1 + kernel_hsize(1);            % first kernel center point that fits acs region            
            kx_end = num_acsX - kernel_hsize(1);       % last kernel center point that fits acs region
            
            % k-space limits for recon:
            Ky_begin = 1 + Ry * kernel_hsize(2);       % first kernel center point that fits acs region
            Ky_end = N_pad(2) - Ry * kernel_hsize(2);  % last kernel center point that fits acs region
            Kx_begin = 1 + kernel_hsize(1);            % first kernel center point that fits acs region
            Kx_end = N_pad(1) - kernel_hsize(1);       % last kernel center point that fits acs region
            
            % count the no. of kernels that fit in acs
            ind = 1;
            for ky = ky_begin : ky_end
                for kx = kx_begin : kx_end
                    ind = ind + 1;
                end
            end
            num_ind = ind;

            Img_recon = zeros([N, num_chan, num_cycle]);
            
            for pc = 1:num_cycle
                kspace_full = kspace_data(:,:,:,pc);
                kspace_sampled = kspace_full .* repmat(mask(:,:,pc), [1,1,num_chan]);
                kspace_acs = single_acs;
                
                % train kernel
                kspace_acs_crop = kspace_acs(1+end/2-num_acsX/2:end/2+num_acsX/2, 1+end/2-num_acsY/2:end/2+num_acsY/2, :);
                
                Rhs = zeros([num_ind, num_chan, Ry-1]);
                Acs = zeros([num_ind, prod(kernel_size) * num_chan]);               
                size(Acs)
                              
                ind = 1;               
                for ky = ky_begin : ky_end
                    for kx = kx_begin : kx_end
                        acs = kspace_acs_crop(kx-kernel_hsize(1):kx+kernel_hsize(1), ky-kernel_hsize(2)*Ry:Ry:ky+kernel_hsize(2)*Ry, :);
                        Acs(ind,:) = acs(:);
                        for ry = 1:Ry-1
                            Rhs(ind,:,ry) = kspace_acs_crop(kx, ky-ry, :);
                        end
                        ind = ind + 1;
                    end
                end
                
                if lambda_tik
                    [u,s,v] = svd(Acs, 'econ');
                    s_inv = diag(s);
                    s_inv = conj(s_inv) ./ (abs(s_inv).^2 + lambda_tik);
                    Acs_inv = v * diag(s_inv) * u';
                end
                
                % estimate kernel weights
                weights = zeros([prod(kernel_size) * num_chan, num_chan, Ry-1]);
                for r = 1:Ry-1
                    disp(['Kernel group : ', num2str(r)])
                    for c = 1:num_chan
                        if ~lambda_tik
                            weights(:,c,r) = Acs \ Rhs(:,c,r);
                        else
                            weights(:,c,r) = Acs_inv * Rhs(:,c,r);
                        end
                    end
                end
                
                % reconstruct undersampled data
                Weights = permute(weights, [2,1,3]);
                kspace_recon = padarray(kspace_sampled, [pad_size, 0]);
                
                for ky = Ky_begin+pc-1 : Ry : Ky_end
                    for kx = Kx_begin : Kx_end
                        data = kspace_recon(kx-kernel_hsize(1):kx+kernel_hsize(1), ky-kernel_hsize(2)*Ry:Ry:ky+kernel_hsize(2)*Ry, :);
                        for ry = 1:Ry-1
                            kspace_recon(kx, ky-ry, :) = Weights(:,:,ry) * data(:);
                        end
                    end
                end
                
                kspace_recon = kspace_recon(1+pad_size(1):end-pad_size(1), 1+pad_size(2):end-pad_size(2), :);
                
                % subsititute sampled & acs data
                kspace_recon = kspace_recon .* repmat((~mask(:,:,pc) ), [1,1,num_chan]) + kspace_sampled .* repmat(mask(:,:,pc), [1,1,num_chan]);
                Img_recon(:,:,:,pc) = ifft2c(kspace_recon);
            end
            
            % combine multiple coil data
            input = mean(rsos(Img_recon, 3),4); % mean in the 4th dimension
            input = imresize(input,[size(Img_recon,1) size(single_RawDate_selected,2)]); 
            input_all(:,:,sl) = input;
        end
        

%% reorder and save to nii

        if sl > 1
            if mod(sl,2) == 0
                % if slice_num is even
                for i = 1:sl/2
                    input_all_reorder(:,:,i*2) = input_all(:,:,i);
                end
                k = sl/2;
                for j = sl/2+1:sl
                    input_all_reorder(:,:,j-k) = input_all(:,:,j);
                    k = k-1;
                end
            else
                % if slice_num is odd
                for i = 1:(sl+1)/2
                    input_all_reorder(:,:,2*i-1) = input_all(:,:,i);
                end
                k = (sl-1)/2;
                for j = (sl+1)/2+1 : sl
                    input_all_reorder(:,:,j-k) = input_all(:,:,j);
                    k = k-1;
                end
            end
        else
            input_all_reorder = input_all;
        end

        size_input_all_reorder = size(input_all_reorder);
        input_all_reorder = crop(input_all_reorder,[size_input_all_reorder(1)/2,size_input_all_reorder(2:end)]);

        mosaic(imrotate(input_all_reorder, 180), 1, sl, 1, ['Meas. ',num2str(meas)], 0.6*genCaxis(input_all_reorder))
        
        origin = [0 0 0];
        datatype = 16; % single float
        nii = make_nii(imrotate(fliplr(input_all_reorder), 90), origin, datatype);
        savefilepath = [pwd,'\recon_nii\',name(1:end-4)];
        filename = [savefilepath,'\GRAPPA_',name(1:end-4),'_',num2str(meas,'%04d')];
        save_nii(nii,[filename,'.nii']);
        
        clear input_all_reorder;
    end
    
    clear Weights weights v u single_struct single_RawDate_selected single_RawDate_all...
          single_RawDate single_acs s_inv s Rhs nii... 
          multi_struct multi_RawData mask...
          kspace_sampled kspace_recon kspace_full kspace_data_all kspace_data kspace_acs_crop...
          kspace_acs input_all input Img_recon Acs_inv Acs 
    close all
    
end
