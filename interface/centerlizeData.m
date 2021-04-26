function [ I_tr, I_te, T_tr, T_te ] = centerlizeData( I_tr, I_te, T_tr, T_te )
% CENTERLIZEDATA 
    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

end

