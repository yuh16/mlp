function fx_drev = Activation_func_drev(fx)
    
    fx_drev = fx .* (1 - fx); %Binary

end