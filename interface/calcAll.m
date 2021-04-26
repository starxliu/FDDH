function [ map, prc, topK ] = calcAll( tselLabel, trainLabel, testInstance, trainInstance, K )
% CALCALL mAP£¬topmAP£¬PR
% input£º
%   testLabel£ºlabel of testing data  numQuery * numLabel
%   trainLabel£ºlabel of training data  n * numLabel
%   testInstance£º hash code of test data, numQuery * bits
%   trainInstance: hash code of training data, n * bits
%   K£ºparameter of topKmap
% output£º
%   map£ºmAP value
%   prc£ºpr curve values
%   topK£ºinclude  map, topkMap£¬topkPre£¬topkRec

  
    testInstance = bitCompact( testInstance >= 0 );
    trainInstance = bitCompact( trainInstance >= 0 );
    
   
    if exist('K', 'var')
        topK = calcMapTopkMapTopkPreTopkRecLabel( tselLabel, trainLabel, testInstance, trainInstance, K );
        map = topK.map;
    else
        map = calcMapTopkMapTopkPreTopkRecLabel( tselLabel, trainLabel, testInstance, trainInstance );
        topK = 0;
    end
    
    prc = calcPreRecRadiusLabel( tselLabel, trainLabel, testInstance, trainInstance );

end

