function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C1 = [.01 .03 .1 .3 1 3 10 30];
sigma2 = [.01 .03 .1 .3 1 3 10 30];

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
min_error=intmax;

for i=1:8
    for j=1:8
        model=svmTrain(X,y,C1(i), @(x1,x2) gaussianKernel(x1,x2,sigma2(j)));
        predictions=svmPredict(model,Xval);
        error=mean(double(predictions ~=yval));
        if(error<min_error) 
            C=C1(i);
            sigma=sigma2(j);
            min_error=error;
        end
    end
end

            
        
    



% =========================================================================

end
