function [dC_dw, dC_db, grad_C] = BLK2_backward_pass(inputs, weights, biases, grad_C)
%cost function?
%weight adjustment

    net_error=biases; %need cell array containing vectors same size as biases 
    
    dC_dw=weights; dC_db=biases;
    for i=1:length(weights)
        dC_dw{i}=ones(size(weights{i}));
        dC_db{i}=ones(size(biases{i}));
    end

    [y_l, z_l]=forward_pass(inputs, weights, biases);
    
    %Get neuron errors
    for l=length(net_error) 
        net_error{l}=grad_C.*arrayfun(@(x) sigmoid_prime(x), z_l{end});
    end
    
    for l=(length(net_error)-1):-1:2
        net_error{l}=transpose(weights{l+1})*net_error{l+1}.*arrayfun(@(x) sigmoid_prime(x), z_l{l});
    end
    l=-1;
    %computing the partial derivatives
    for l=length(net_error):-1:2
        dC_dw{l}=dC_dw{l}+net_error{l}*y_l{l-1}';
        dC_db{l}=dC_db{l}+net_error{l};
        
        %fprintf('Layer %d', l);
%         for j = 1:length(biases{l})%current layer neuron indices
%             for k = 1:length(biases{l-1}) %previous layer neuron indices
%                 %fprintf('Layer %d', l);
%                 dC_dw{l}(j,k)=dC_dw{l}(j,k)+y_l{l-1}(k)*net_error{l}(j);
%                 %dC_dw{2}
%                 dC_db{l}(j)=dC_db{l}(j)+net_error{l}(j);
%             end
%         end
        
    end

end

