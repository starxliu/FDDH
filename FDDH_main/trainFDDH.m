%If you use our codes, we are appreciated if you appropriately cite our work.
%	X. Liu*, X.Z. Wang, Y.M. Cheung, FDDH: Fast Discriminative  Discrete  Hashing for Large-Scale Cross-Modal Retrieval. IEEE Transactions on Neural Networks and Learning Systems, in press.

clear
warning off
addpath('../data');
addpath('../interface');


datasets = {'WikiData'};

data_N = length(datasets);
K = 50;
runtimes = 1;
gamma = 1e-4;

for data_i = 1: data_N
    clearvars -except datasets data_N K runtimes data_i gamma
    load([datasets{data_i},'.mat']);
    
    num = size(I_tr,1);
    
    switch datasets{data_i}
         case 'WikiData'
            ind = [5 10 30 50 100 250:250:2000 num];
            anchorNum = 800;
            gamma1 =  1e4;
            mu = 1e-3;
            theta = 1e-3;
        case 'MIRFlickr25k'   % handcrafted feature for image
            ind = [10 50 100 300 500 1000:1000:10000 num];
            anchorNum = 800;
            gamma1 =  1e5;
            mu = 1e-3;
            theta = 1e0;
    end
    
    c = size(L_tr,2);
    globalBits = [32, 64, 128];
    bit_N = length(globalBits);
    [ I_tr, I_te, T_tr, T_te] = centerlizeData(double(I_tr), double(I_te), double(T_tr),double(T_te));
    
    tic
    n = size(I_tr, 1);
    anchorIndex = sort(randperm(n, anchorNum));
    [Ktr1, Ktr2, Kte1, Kte2, Kanchor1, Kanchor2] = kernelTrans(I_tr, T_tr, I_te, T_te, anchorIndex);
    
    Kernel_Time = toc;
    nt = size(I_te,1);
    % different length of hash code
    for bit_i = 1: bit_N
        bit = globalBits(bit_i);
        %  repeat more time
        for run_i = 1: runtimes
            tic
            
            [ S ,L,iter, loss ] = solveFDDH( L_tr', Ktr1', Ktr2',mu, theta, gamma1, bit);
            
            Time(bit_i, run_i) = toc;
            Iter(bit_i, run_i) = iter;
            Loss(bit_i, run_i) = loss;
            [ P1, P2] = solveUnseen_LP( S, Ktr1', Ktr2', gamma);
            HI_te = sign(P1 * Kte1');   % bits * n
            HT_te = sign(P2 * Kte2');   % bits * n
            
            %% cross modal retrieval
            [MAP(bit_i, 1), FDDH_prc(bit_i, 1),FDDH_topk(bit_i, 1)] = calcAll(L_te, L_tr, HI_te', S', ind);
            [MAP(bit_i, 2), FDDH_prc(bit_i, 2),FDDH_topk(bit_i, 2)] = calcAll(L_te, L_tr, HT_te', S', ind);
            
            top50_map(bit_i, 1) = FDDH_topk(bit_i, 1).topkMap(ind==50);
            top50_map(bit_i, 2) = FDDH_topk(bit_i, 2).topkMap(ind==50);
            
            fprintf('%s, bits: %d, MAPI->T %.4f, MAPT->I %.4f\n', datasets{data_i}, bit, MAP(bit_i, 1), MAP(bit_i, 2));
            
            fprintf('%s, bits: %d, done!\n', datasets{data_i}, bit);
            
        end
    end
    
    dirname = '../result/FDDH_result';
    if ~exist(dirname, 'dir')
        mkdir(dirname);
    end
    name_R = [dirname,'/FDDH_',datasets{data_i}, '.mat'];
    save(name_R, 'MAP');
    name = [dirname,'/FDDH_',datasets{data_i},'_PRC','.mat'];
    save(name,'FDDH_prc','FDDH_topk');
    name_N = [dirname, '/FDDH_Time_Iter_Loss_', datasets{data_i}, '.mat'];
    save(name_N, 'Time', 'Kernel_Time','Iter', 'Loss');
end