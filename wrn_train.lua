--[[
1. Describe the app
1.1 Provide a set of parameters to an object
1.2 Launch the traininer object and wait for results

2. Gather requirements
2.1 Replace the state every N epochs
2.2 Save the log
2.3 Show the chart with error

3. Identify the main objects
3.1 TrainerDirector Object
3.2 Trainer Builder
3.2.1 Abstract Builder Protocol
3.3 Trainer
3.4 Net Object
3.5 DataProvider Object
3.6 Abstract Decorator Object
3.4 NetOutputStorage Decorator Object
3.5 Logging Decorator Object
3.6 Visualization Decorator Object (Chart.. although can be different)
3.7 Decoration Builder Protocol
3.7.1 Decoration Abstract Builder Protocol
3.8 Director for Decorator Object

4. Identify relationships between the objects
4.1 TrainderDirector assembles Trainer Object based on builder methods
4.2 Builder encapsulates an object and adds Net, DataProvider to a Trainer object
4.3 Trainer object provides a method to launch training with certain parameters
]]--
if not torch then require 'torch' end
--if not cudnn then require 'cudnn' end
--if not cltorch then require 'cltorch' end
--if not nn then require 'clnn' end
if not Mediator or not AbstractObject then require 'trainer/Mediator/Mediator' end --provides class AbstractObject and Mediator
local TrainerDirector = require 'trainer/TrainerDirector'
local TrainerBuilder = require 'trainer/TrainerBuilder'
local Trainer = require 'trainer/trainer_v4' --local Trainer = require 'trainer/Trainer'
local Debuger = require 'trainer/Mediator/Debugger'
local Logger = require 'trainer/Mediator/Logger'
local NetSave = require 'trainer/Mediator/NetSave'
--
local dataLoader_opt = {
  gen = 'Results',
  dataset_path = '/home/ubuntu/data/home/ubuntu/CACD2000',
  mean_std_path = 'Results/norm.t7',
  train_descriptor = 'Results/cacd_train.t7', 
  val_descriptor = 'Results/cacd_val.t7', 
  scale = 256,
  --flip = {vertircal = nil},
  --crop = 128,
  --fivecrop = 128,
  batch_size = 58,
  n_batches = 13,
  multithreading = true
}
local netLoader_opt = {
  file_name = 'wrn27',
  save_path = 'Results',
  pretrainedNet_mode = nil,
  dropout = 0.5, 
  depth = 22,
  num_classes = 1997,
  widen_factor = 2,
  imageSize = dataLoader_opt.scale,
  pretrainedNet_path = 'Results/wrn27.t7',
  --pretrainedOpt_path = 'Results/ResNet50_drout05_224x224_opt.t7',
  pretrainedNet_mode = 'binary'
}
local training_opt = {
  criterion = nn.CrossEntropyCriterion(),
  epochs = 100,
  epoch_step = 4,
  save_net_epochs = 1,
  GPU_Index = 1,
  GPU_type = 'cudnn',
  --[[learningRate = 1e-4,
  optimMethod = 'nag',
  momentum = 0.9,
  weightDecay = 0.05,
  learningRateDecayRatio = 0.1,]]--
  optimMethod = 'adam',
  learningRate = 1e-3,
  lossLogFrequency = 50,
  confusionLogFrequency = nil,
  printingFrequency = 50,
  testingFrequency = 50,
  start_iteration = nil
}
--
local trainerDirector = TrainerDirector(TrainerBuilder(Trainer(netLoader_opt, dataLoader_opt, training_opt, Mediator)))
local trainer = trainerDirector:assemble()
--Add trainer to the list of objects Mediator will manage for :send() and :recieve() methods
--add trainer to the moderator
Mediator:addObj(trainer)
--add debugger to the moderator
Mediator:addObj(Debuger({printingFrequency = training_opt.printingFrequency}, Mediator))
--add debugger to the moderator
Mediator:addObj(Logger({
      confusion_path = 'Results/confusion.txt',
      log_path = 'Results/training.log',
      confusionLogFrequency = nil,
      classList = trainer.classList,
      plotFlag = training_opt.testingFrequency
      }, Mediator))
--add netsaver to the moderator
Mediator:addObj(NetSave(netLoader_opt.save_path, netLoader_opt.file_name, Mediator))
--Once the Director is done, free up the memory
trainerDirector = nil
--let's check that the net was properly constructed
trainer:showNet()
--launch the training cycle
trainer:launch(training_opt.epochs)
--[[Things to add:
tesh batch normalization as a way to avoid preprocessing data
add input size auto detection to the network
]]--
