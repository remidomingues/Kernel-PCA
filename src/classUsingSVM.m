function err = classUsingSVM(Y,Z,train_index,test_index)

%LABELS = zeros(length(test_index),10);
%PROB = zeros(length(test_index),10);

%for ii = 1:10
%    model = svmtrain(double((train_index==ii)'),Y');
%    [label, ~, prob] = svmpredict(double((test_index==ii)'), Z', model);
%    LABELS(find(label==1),ii) = ii;
%    PROBS(:,ii) = prob(:);
%end

model = svmtrain(train_index',Y','-t 0' );
[label, ~, ~] = svmpredict(test_index', Z', model);
err = sum(label'~=test_index)/length(label);

end