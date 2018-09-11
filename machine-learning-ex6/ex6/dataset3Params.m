function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_choice = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_choice = [0.01,0.03,0.1,0.3,1,3,10,30];
err = zeros(length(C_choice),length(sigma_choice));
for i=1:length(C_choice),
    C_temp = C_choice(i);
    for j=1:length(sigma_choice),
        sigma_temp = sigma_choice(j);
        model = svmTrain(X,y,C_temp,@(x1,x2)gaussianKernel(x1,x2,sigma_temp));
        err(i,j) = mean(svmPredict(model,Xval) ~= yval);
    end;
end;
[value_col,index_ln] = min(err);
[value,index] = min(value_col);
C = C_choice(index_ln(index));
sigma = sigma_choice(index);

% =========================================================================

end
