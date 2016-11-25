--[[
Logger - saves log for loss and confuion matrix as well as provides online vizualization
Required Objects:
Mediator object needed for initialization of the debugger

Related Objects
Parrent: AbstractObj (part of mediator.lua module)

Required Functions:
:__init(mediator) init super with mediator
:send(message) method inherited from AbstrackObj class
:recieve(msg) stores recieved loss value into a tex file as well as adds the data to confusion matrix
:saveConfusionMatrix()
]]--
if not torch then require 'torch' end
if not paths then require 'paths' end

if not image then require 'image' end

if not optim then require 'optim' end
if not class then class = require 'class' end
if not Mediator or not AbstractObject then require 'trainer/Mediator/Mediator' end --provides class AbstractObject and Mediator
--
local Logger = class('Logger', 'AbstractObject')
function Logger:__init(opt, mediator)
  AbstractObject.__init(self, mediator)
  self.plotFlag = opt.plotFlag or false
  self.filePath = opt.confusion_path or 'confusion.t7'
  self.confusionLogFrequency = opt.confusionLogFrequency
  if self.confusionLogFrequency then 
    self.confusion = optim.ConfusionMatrix(opt.classList)
    self.confusion:zero()
  end --self.classList will be provided by TrainerBuilder:createDataSource(...) - consider changing the code to remove this dependency
  self.logger = optim.Logger(opt.log_path or 'log.t7')
  self.logger:setNames{'Training error', 'Val error'}
  self.logger:style{'+-', '--'}
end
function Logger:receive(msg)
  --msg should contain .key and .value
  local availableKeys = {'log'}
  local rightKeyFlag = false
  for _, i in pairs(availableKeys) do
    if msg.key == i then rightKeyFlag = true end
  end
  if rightKeyFlag then
    self.logger:add{msg.value.loss, msg.value.validation_loss or 0}
    if self.plotFlag and (msg.value.iter > 1) and (msg.value.iter % self.plotFlag == 0) then
      self.logger:plot()
    end
    if self.confusionLogFrequency then
      self.confusion:batchAdd(table.unpack(msg.value.confusion_data))
      if (msg.value.iter >= self.confusionLogFrequency) and (msg.value.iter % self.confusionLogFrequency == 0) then
        self:saveConfusionMatrix()
      end
    end
  end
end
function Logger:saveConfusionMatrix()
  torch.save(self.filePath, tostring(self.confusion), 'ascii')
  print("Confusion matrix was saved to " .. self.filePath)
  print("Confusion matrix results are:")
end
return Logger
