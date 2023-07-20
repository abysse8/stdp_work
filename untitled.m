OutNeuron1_15 = load('export15.mat', 'OutNeuron1');
OutNeuron1_15 = getfield(OutNeuron1_15, 'OutNeuron1');

OutNeuron1_18 = load('export18.mat', 'OutNeuron1');
OutNeuron1_18 = getfield(OutNeuron1_18, 'OutNeuron1');

OutNeuron1_22 = load('export22.mat', 'OutNeuron1');
OutNeuron1_22 = getfield(OutNeuron1_22, 'OutNeuron1');

OutNeuron1_26 = load('export26.mat', 'OutNeuron1');
OutNeuron1_26 = getfield(OutNeuron1_26, 'OutNeuron1');

OutNeuron1_30 = load('export30.mat', 'OutNeuron1');
OutNeuron1_30 = getfield(OutNeuron1_30, 'OutNeuron1');

OutNeuron1_34 = load('export34.mat', 'OutNeuron1');
OutNeuron1_34 = getfield(OutNeuron1_34, 'OutNeuron1');

all_trials = {'OutNeuron1_15', 'OutNeuron1_18', 'OutNeuron1_22', 'OutNeuron1_26', 'OutNeuron1_30', 'OutNeuron1_34'};
%% 
time = 10;
timestep = 1/1000; 
hold on;

for jjj= 1:length(all_trials)
    input = zeros(1, time/timestep);
    trial = eval(all_trials{jjj});
    for iii=1:length(trial)
        input(1, round(trial(iii)/timestep)) = 1;
    end
    subplot(length(all_trials),1,jjj);
    plot(input(900:1600))
end
%%
result = zeros()
for freq = 0.01:0.01:10%length(example)
        dummy_vector = zeros(1,length(example));
        for iii = 1:length(dummy_vector)
            if mod(iii+r,t) == 0
                dummy_vector(iii) = 1;
            end
        end
        result = [result t*sum(dummy_vector.*example)/length(example)];
end
plot(result)

