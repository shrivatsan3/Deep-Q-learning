############
# Question 3
############

using DMUStudent.HW5: HW5, mc
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
using VegaLite
using ElectronDisplay: electrondisplay
using DataFrames: DataFrame
using StaticArrays
using Random: randperm, shuffle!

mutable struct DeepQL
	Q
    FixQ
    BestQ
    γ 
	ϵ 
    act
	act_ind 
    buffer
    limit
    decay
    epoch
    episodes
    ravg
    steps
end

function epsilon_greedy(s)
    
    if rand() < DQL.ϵ
        
        if sign(s[2]) == 1 # Take action in the direction of your velocity
            return 3
        elseif sign(s[2]) == -1
            return 1
        elseif sign(s[1]) == -1
            return 3
        else
            return 1
        end
    
    else    
        return argmax(DQL.Q(s))
    end
end	


function (DQL::DeepQL)(env)
        # use epsilon greedy to select an action
        # Start with some state
        s = observe(env) 
    
        # use epsilon greedy to select an action
        ai = epsilon_greedy(s) 
        #a = softmax(s)
        # A new state and reward will be generated based on action taken
        r = act!(env, DQL.act[ai]) 
        DQL.steps+=1
        sp = observe(env)
        
        # Keep repeating the process of generating new state and reward till terminal state is reached
        # and update Q value averages
        index = 0
        DQL.limit = DQL.γ
        while ((!terminated(env)) && (DQL.limit>0.005))
    
            index +=1
            #Add to experience replays
            experience_tuple = (s, ai, r, sp, false)
            push!(DQL.buffer,experience_tuple)
            #@show length(DQL.buffer)
            #println("here")
            s = sp
            ai = epsilon_greedy(s) 
            r = act!(env, DQL.act[ai])
            DQL.steps+=1
            sp = observe(env)
            DQL.limit*=DQL.γ
            #@show DQL.limit
            #println("exploring")
        end
        
        if DQL.limit > 0.005    
            #Add to experience replay
            experience_tuple = (s, ai, r, sp, true)
            push!(DQL.buffer,experience_tuple) 
        end
         
end
    
# Since the mc environment has a continuous action space and DQN uses a discrete action space, you can choose a subset of the actions to use and create an environment with an overridden action space:
env = QuickWrapper(mc, actions=[-5.0, 0.0, 5.0])

# Initialize DQL learning object
DQL = DeepQL(   Chain(Dense(2, 64, relu),Dense(64, length(actions(env)))), # Q is a chain object
                Chain(Dense(2, 64, relu),Dense(64, length(actions(env)))), # Q is a chain object
                Chain(Dense(2, 64, relu),Dense(64, length(actions(env)))), # Fix Q for regression
                0.99, #γ
                1, #ϵ  
                actions(env),
                [i for i in 1:length(actions(env))], # action indices every action
                [], #Empty array for buffer
                0.99, #limit
                0, #epsilon decay
                200, #epochs
                20,#episodes
                0,#ravg
                0) # number of steps in env 
# The following are some basic components needed for DQN
# create your loss function for Q training here

function NeuralNet(DQL::DeepQL)
    data = rand(DQL.buffer,10000)

    function loss(s, a_ind, r, sp, done)
        return (r + (!done)*DQL.γ*(maximum(DQL.FixQ(sp))) - DQL.Q(s)[a_ind] )^2 # this is not correct! you need to replace it with the true Q-learning loss function
        # make sure to take care of cases when the problem has terminated correctly
    end
    

    DQL.FixQ = deepcopy(DQL.Q)
    for t = 1:1
        #@show t
        global test = 0
        Flux.Optimise.train!(loss, Flux.params(DQL.Q), data, ADAM(0.01))
    end
end
# Evaluating learned policy - Evaluation episode

StepCount = [] # store number of steps
AvgRet = [] # store avg returns

function  evaluate(DQL)
	rsum = 0
	for ii = 1:1000
		reset!(env)
		DQL.limit = DQL.γ
        while ((!terminated(env)) && (DQL.limit>0.005))
        	s = observe(env)
			eval_act = argmax(DQL.Q(s))
			rsum += (DQL.γ^ii)*act!(env, DQL.act[eval_act])
            DQL.limit *= DQL.γ
		end
	end
	return rsum/1000
	
end

for i in 1:DQL.epoch
	@show i
    
    for ep = 1:DQL.episodes
        DQL.decay +=1
        DQL.ϵ = max(0.1,1-DQL.decay/(DQL.epoch*DQL.episodes))
        reset!(env)
        DQL(env)
        @show (i,ep)
    end
	@show length(DQL.buffer)
    
    NeuralNet(DQL)

    @show rtemp = evaluate(DQL)
    push!(StepCount, DQL.steps)
    push!(AvgRet,rtemp)
    if rtemp > DQL.ravg
        DQL.ravg = rtemp
        DQL.BestQ = deepcopy(DQL.Q)
    end
    
end
#HW5.evaluate(s->actions(env)[argmax(DQL.BestQ(s))],"shkc2154@colorado.edu")
