--[[
Existing training process description:
  1. initialize a criterion -- should be done at initialization
  2. initialize a confusion matrix -- should be done at initialization point
  3. initialize w, wg — should be done at initialization point
  4. create feval function
  5. initialize losses table
  6. initialize iterNumber
  7. announce initiation of the training
  8. create learning_rate_step function
  9. create printing function
  10. create loss logging function

Object Oriented Analysis for Training process:
Requirements:
1. Provide trained network and debugging information

Identify objects:
1. trainer - execute training to create a network and debugging information
2. debugger - prints out timing (function 1) and debugging (function 2 and 3) information with a given frequency (parameter)
3. netExporter - save trained net after given number of iterations (parameter)

Describe every object:
Trainer:
1. Main responsibility - train a network
1.1 initTrainingVariables
1.2 feval
2. Secondary responsibilities:
2.1 Track losses
2.2 Modify learning rate (1.given number of steps; 2. later look into modifying learning rate based on descend curve)

Related class
1. DataProvider - Provides the data
2. Net - Provides Neural Network
3. [to be built] SaveWeights - saves current state of the network weights and options settings
4. [to be built] Logger - stores results of the log of the loss in a disk file
5. [to be built] Visualizer - builts a chart of the loss history
]]--

if not torch then require 'torch' end
if not optim then require 'optim' end
if not nn then require 'nn' end
--local optnet = require 'optnet'
if not class then class = require 'class' end
local GPUTester = require 'GPUTest/GPUTester'

local Trainer = class('Trainer', 'AbstractObject')
function Trainer:__init(net_opt, dataLoader_opt, training_opt, mediator)

  AbstractObject.__init(self, mediator)
  self.GPU = GPUTester(training_opt.GPU_Index, training_opt.GPU_type)
  self.net_opt = net_opt
  self.dataLoader_opt = dataLoader_opt
  self.training_opt = training_opt
  --variables to be provided through assemble method of director/builder
  self.net = nil
  self.commander = nil
  self.dataProvider = nil
  self.validation_dataProvider = nil
  self.batchSize = nil
  self.training_set_size = nil
  self.classList = nil
  self.w, self.w_gradient = nil, nil
  --variables to be provided by initTrainVar() method
  self.confusion = nil
  self.iterations = 5
  self.save_iterations = 1000
  self.avg_loss = 0.0
end
function Trainer:receive(msg)
  print("Error: an object of Trainer type received a message: ")
  print(msg)
  assert(false, "Trainer is not designed to respond to messages. Review the code")
end
function Trainer:initTrainVar(epochs)
  --setting up anxiliary variables for debugging process
  self.iterations = epochs * self.training_set_size / self.batchSize
  local save_net_epochs = self.training_opt.save_net_epochs or 5
  self.save_iterations = self.training_set_size / self.batchSize * save_net_epochs
  self.w, self.w_gradient = self.net:getParameters()
end
--
function Trainer:getData()
  local data = self.dataProvider:get()
  local targets = self.GPU:convert(data.class)
  data = self.GPU:convert(data.data)
  collectgarbage()
  return data, targets
end
function Trainer:getValidationData()
  local data = self.validation_dataProvider:get()
  local targets = self.GPU:convert(data.class)
  data = self.GPU:convert(data.data)
  collectgarbage()
  return data, targets
end
function Trainer:showNet()
  print("Provided net topology presented below:")
  --print(tostring(self.net))
  print(self.net)
end
function Trainer:learningRateUpdate(iteration)
  if self.training_opt.epoch_step and self.training_opt.learningRateDecayRatio then 
    local step_size = math.ceil(self.training_opt.epoch_step * self.training_set_size / self.batchSize)
    if iteration % step_size == 0 and self.training_opt.learningRate > 1e-7 then
      self.training_opt.learningRate = self.training_opt.learningRate * self.training_opt.learningRateDecayRatio
      self.training_opt.epoch_step = self.training_opt.epoch_step * 3
    end
  end
end
function Trainer:output(i, minibatch_loss, test_loss, target)
  self:learningRateUpdate(i)
  if i % self.training_opt.printingFrequency == 0 then
    self.avg_loss = self.avg_loss / self.training_opt.printingFrequency
    self:send({key = "debug", value = {
          loss = self.avg_loss,
          validation_loss = test_loss,
          iter = i,
          lr = self.training_opt.learningRate
        }
      })
    if minibatch_loss > 1.0 or test_loss > 1.0 then
      print("error is too large to log.. training loss: " .. minibatch_loss .. " and test loss: " .. test_loss)
    else
      self:send({key = "log", value = {
            loss = self.avg_loss,
            validation_loss = test_loss,
            iter = i,
            confusion_data = {self.net.output, target}
          }
        })
    end
    self.avg_loss = 0.0
  else
    self.avg_loss = self.avg_loss + minibatch_loss
  end
  self.training_opt.start_iteration = i
  if i % math.floor(self.save_iterations) == 0 then
    self:send({key = "save", value = {net = self.net, opt = self.training_opt}})
  end
end
function Trainer:learn(data, targets)
  self.w_gradient:zero() -- reset gradient as they accumulate for batch method, and if you don’t  reset it, the network will blow up
  -- evaluate function for complete mini batch
  local loss = self.training_opt.criterion:forward(self.net:forward(data), targets)
  self.net:backward(data, self.training_opt.criterion:backward(self.net.output, targets))
  local feval = function(n_w)
    if self.w ~= n_w then self.w:copy(n_w) end
    return loss, self.w_gradient
  end
  optim[self.training_opt.optimMethod](feval, self.w, self.net_opt)
  collectgarbage()
  return (1 - math.exp(-loss))
end
function Trainer:test(data, targets)
  self.net:evaluate()
  local loss = self.training_opt.criterion:forward(self.net:forward(data), targets)
  self.net:training()
  return (1 - math.exp(-loss))
end
function Trainer:launch(epochs)
  --initialize training variables
  local l, tl = nil, nil
  self:initTrainVar(epochs)
  print("starting training process for "..self.iterations.." iterations or "..epochs.." epochs")
  --start the timer for the training
  self:send({key = "start", value = 1})
  --train a network using given optimization method for a given number of epochs
  local start = self.training_opt.start_iteration or 1
  for i = start, self.iterations do
    local data, targets = self:getData()
    --calculate loss
    l = self:learn(data, targets)
    --let's test our net on a new (validation) dataset with a sertain interval
    if (i > 0) and (i % self.training_opt.testingFrequency == 0) then
      local data, targets = self:getValidationData()
      tl = self:test(data, targets)
    end
    tl = tl or l
    self:output(i, l, tl, targets)
  end
end
--
return Trainer
