% 布谷鸟算法
%     fun_name 寻优函数
%     nd       需要寻优的参数个数
%     Lb       鸟窝范围的下限
%     Ub       鸟窝范围的上限
function [bestnest, fmin] = Cuckoo(fun_name, nd, Lb, Ub)

    global n;
    
    n = 15; % 种群数量
    pa = 0.25;  % 布谷鸟蛋被发现的概率
    Tol = 1e-6; % 阈值循环的精度设置
    maxcycles = 1000;
    nest=Lb+(Ub-Lb).*rand(n,nd);% 鸟窝的初始值
    fitness=10e10*ones(n,1);%寻优域的范围设置

    %初始化巢穴
    [fmin,bestnest,nest,fitness]=get_best_nest(nest,nest,fitness,fun_name);


    %% 循环计算
    for count = 1:maxcycles

        % 莱维飞行，依据 原解与最优解 产生新解
        new_nest = get_cuckoos( nest, bestnest, Lb, Ub );

        %比较原解与新解产生 较优解:nest 
        [fnew, best, nest, fitness] = get_best_nest( nest,new_nest, fitness,fun_name); %#ok<ASGLU>

        % 布谷鸟蛋被发现
        new_nest=empty_nests(nest,Lb,Ub,pa) ;

        % 在更新后的鸟窝数据中再次找到最佳巢穴
        [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,fun_name);


        % 找到最终最优值
        if fnew<fmin%如果新鸟窝与目标鸟窝之间的的距离更近(即误差精度更小更准确)
            fmin=fnew;%则用新的鸟窝的值替换原fmin中的数据
            bestnest=best;%并且把此鸟窝下的各个参数的最优解赋给bestnest
        end
        
        % 达到误差允许范围，结束
        if fmin < Tol
            break;
        end
        
    end
    
% 结束
end




%% 
% 最小的解 bestnest
% 最小值 fmin
% 误差 Tol






%% 布谷鸟算法莱维飞行
function nest=get_cuckoos(nest,best,Lb,Ub)

    global n;

    beta=3/2;
    aplha = 0.01; %飞行步长缩放常量

    %sigma这个数约等于0.7
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

    for j=1:n

        s=nest(j,:); %取出第  j  个个体

        %randn函数产生均值为0,方差σ^2 = 1的正态分布的随机数或矩阵的函数
        u=randn(size(s))*sigma;
        v=randn(size(s));
        step=u./abs(v).^(1/beta);
        stepsize=aplha*step.*(s-best);% 步长

        s=s+stepsize.*randn(size(s));

        nest(j,:)=simplebounds(s,Lb,Ub);%获得交替比较上下限的值
    end

end




%% 输入原解与新解以及原解的输出值，寻找新解中最优的代替原解，更新输出值
function [fmin,best,nest,fitness]=get_best_nest(nest,newnest,fitness,fun_name)

    global n;

    for j=1:n  
        fnew=feval(fun_name, (newnest(j,:)) ); % 计算新解的值
        
        if fnew<=fitness(j) % 优质解替换
            fitness(j)=fnew; % 更新解
            nest(j,:)=newnest(j,:); % 更新输出值
        end
    end
    
    % 记录此刻种群中最优值的一个解
    [fmin,K]=min(fitness) ;
    best=nest(K,:);%获得当前最好的巢穴的值
 
end




    %% 布谷鸟蛋被发现过程
    function new_nest=empty_nests(nest,Lb,Ub,pa)

    global n;

    K=rand(size(nest))>pa;%如果产生的随机数大于pa的值，就令K等于该值

    % 解集随机行走的步长
    stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
    
    % 构造新解集
    new_nest=nest+stepsize.*K;
    
    % 解集是否在范围内判断
    for j=1:n
        s=new_nest(j,:);
        new_nest(j,:)=simplebounds(s,Lb,Ub);
    end
    
end




%% 判断解是否在取值范围内
function s=simplebounds(s,Lb,Ub)

ns_tmp=s;%赋值

% 下限
I=ns_tmp<Lb;
ns_tmp(I)=Lb(I);

% 上限
J=ns_tmp>Ub;
ns_tmp(J)=Ub(J);

s=ns_tmp;
end
