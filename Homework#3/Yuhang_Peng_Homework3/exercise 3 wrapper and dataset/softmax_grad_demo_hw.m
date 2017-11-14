function softmax_grad_demo_hw()
% softmax_grad_demo_hw - when complete reproduces Figure 4.3 from Chapter 4
% of the text

%%% load data 
[X,y] = load_data();

%%% run gradient descent 
w = softmax_gradient_descent(X,y);

%%% plot everything, pts and lines %%%
plot_all(X',y,w);


%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%
%%% gradient descent function for softmax cost/logistic regression %%%
function w = softmax_gradient_descent(X,y)
    %%% initialize w0 and make step length %%%
    X = [ones(size(X,1),1) X]';  % use compact notation
    w = randn(3,1);              % random initial point
    alpha = 10^-2;               % fixed steplength for all iterations
    
    % Initializations 
    iter = 1;
    max_its = 30000;
    grad = 1;
    
    while  norm(grad) > 10^-12 && iter < max_its
        % compute gradient
        grad =  ;           % YOUR CODE GOES HERE
        w = w - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(X,y,w)
    red = [1 0 .4];
    blue =  [ 0 .4 1];

    % plot points 
    ind = find(y == 1);
    scatter(X(1,ind),X(2,ind),'Linewidth',2,'Markeredgecolor',blue,'markerFacecolor','none');
    hold on
    ind = find(y == -1);
    scatter(X(1,ind),X(2,ind),'Linewidth',2,'Markeredgecolor',red,'markerFacecolor','none');
    hold on

    % plot separator
    s =[0:0.01:1 ];
    plot (s,(-w(1)-w(2)*s)/w(3),'m','linewidth',2);
    
    % clean up plot and add info labels
    set(gcf,'color','w');
    axis square
    box off
    axis([0 1 0 1])
    xlabel('x_1','Fontsize',14)
    ylabel('x_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)
end

%%% loads data %%%
function [X,y] = load_data()
    data = csvread('imbalanced_2class.csv');
    X = data(:,1:end-1);
    y = data(:,end);
end

end
