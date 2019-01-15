function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


C = 0;
sigma = 0; 
C_val = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_val = [0.01;0.03;0.1;0.3;1;3;10;30];


min = inf;
for i = 1:8
  for j = 1:8
    model = svmTrain(X, y,C_val(i), @(x1, x2) gaussianKernel(x1, x2, sigma_val(j)));
    e = mean(double(svmPredict(model, Xval) ~= yval));
    if(e < min)
      min = e; 
      C = C_val(i); 
      sigma = sigma_val(j); 
     endif
  endfor
endfor





% =========================================================================

end
