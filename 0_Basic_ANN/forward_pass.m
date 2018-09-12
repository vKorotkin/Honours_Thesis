function [y,z,output]=forward_pass(x, weights, biases)
    y=biases; z=biases; %need shape for arrays
    
    %y: activations for each neuron. last cell array element empty, no
    %nonlinearity at output
    %z: weighted inputs for each neuron. first cell array element empty, no
    %"weighted input" at input
    
    %l for layers
    for l=1
        y{l}=x; %have an output but no weighted input
        z{l}=x;
    end
    
    for l=2:(length(weights)-1) %mult w/ weight matrix, nonlinearity
        z{l}=weights{l}*y{l-1}+biases{l};
        y{l}=arrayfun(@(inp) sigmoid(inp), z{l});
    end
    
    for l=length(weights) %no nonlinearity at last step
        z{l}=weights{l}*y{l-1}+biases{l};
        y{l}=z{l};
        output=y{l};
    end
    
end

