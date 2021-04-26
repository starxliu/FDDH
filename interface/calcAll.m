function [ map, prc, topK ] = calcAll( tselLabel, trainLabel, testInstance, trainInstance, K )
% CALCALL mAP��topmAP��PR
% input��
%   testLabel��label of testing data  numQuery * numLabel
%   trainLabel��label of training data  n * numLabel
%   testInstance�� hash code of test data, numQuery * bits
%   trainInstance: hash code of training data, n * bits
%   K��parameter of topKmap
% output��
%   map��mAP value
%   prc��pr curve values
%   topK��include  map, topkMap��topkPre��topkRec

  
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

