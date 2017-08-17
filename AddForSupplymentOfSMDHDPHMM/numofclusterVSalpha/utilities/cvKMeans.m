function [labels, varargout] = cvKMeans(X, k, varargin)
%cvKMeans Matlab wrapper for kmeans of OpenCV
%   labels = cvKMeans(X, k)
%
%   [labels, centers] = cvKMeans(X, k)
%
%   [labels, centers] = cvKMeans(X, k, maxIter)
%
%   [labels, centers] = cvKMeans(X, k, maxIter, epsilon)
%
%   [labels, centers] = cvKMeans(X, k, maxIter, epsilon, attempts)


narginchk(2, 5);
nargoutchk(1, 2);

switch nargin
    case 2
        if nargout > 1
            [labels, varargout{1}] = OpenCVKMeans4MEX(X, k);
        else
            labels = OpenCVKMeans4MEX(X, k);
        end
    case 3
        if nargout > 1
            [labels, varargout{1}] = OpenCVKMeans4MEX(X, k, varargin{1});
        else
            labels = OpenCVKMeans4MEX(X, k, varargin{1});
        end
    case 4
        if nargout > 1
            [labels, varargout{1}] = OpenCVKMeans4MEX(X, k, varargin{1}, varargin{2});
        else
            labels = OpenCVKMeans4MEX(X, k, varargin{1}, varargin{2});
        end
    case 5
        if nargout > 1
            [labels, varargout{1}] = OpenCVKMeans4MEX(X, k, varargin{1}, varargin{2}, varargin{3});
        else
            labels = OpenCVKMeans4MEX(X, k, varargin{1}, varargin{2}, varargin{3});
        end
end