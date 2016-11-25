--[[
Debugger - prints out timing (function 1) and debugging (function 2 and 3) information with a given frequency (parameter)
Required Objects:
Mediator object needed for initialization of the debugger

Related Objects
Parrent: AbstractObj (part of mediator.lua module)

Required Functions:
:__init(mediator) init super with mediator
:send(message) method inherited from AbstrackObj class
:recieve(msg) stores recieved loss value and :startsTimer(), :checksTimer(), and/or :savesData() depending on the msg.key
:startTimer()
:checkTimer()
:saveData()
:terminateTimer()
]]--
if not torch then require 'torch' end
if not paths then require 'paths' end
if not optim then require 'optim' end
if not class then class = require 'class' end
if not Mediator or not AbstractObject then require 'trainer/Mediator/Mediator' end --provides class AbstractObject and Mediator
--
local Debugger = class('Debugger', 'AbstractObject')
function Debugger:__init(opt, mediator)
  self.printingFrequency = opt.printingFrequency or 5
  AbstractObject.__init(self, mediator)
  self.timer_value = nil
  self.timer_lastCall = 0.0
end
function Debugger:receive(msg)
  --msg should contain .key and .value
  --stores recieved loss value and tracks the time(:timer()), logs the loss (:logLoss()), and/or :savesData() depending on the msg.key
  local availableKeys = {'debug', 'start'}
  local rightKeyFlag = false
  for _, i in pairs(availableKeys) do
    if msg.key == i then rightKeyFlag = true end
  end
  if rightKeyFlag then
    if msg.key == 'start' then
      self:timer()
    elseif msg.value then
      self:timer(msg.value.iter, msg.value.lr, msg.value.loss, msg.value.validation_loss or 0)
    end
  end
end
--
function Debugger:timer(iter, lr, loss, validation_loss)
  if not self.timer_value then
    self.timer_value = torch.Timer()
    print("Starting the timer")
  else
    assert(loss, "Error: self.log was not provided")
    assert(loss>0, "Error: self.log is empty (length = 0)")
    local timeElapsed = self.timer_value:time().real
    local cycleTime = timeElapsed - self.timer_lastCall
    local cycleTime_M = cycleTime/60
    local cycleTime_S = cycleTime % 60
    local timeElapsed_H = timeElapsed/3600
    local timeElapsed_M = (timeElapsed/60) % 60
    local timeElapsed_S = timeElapsed % 60
    print(string.format(
        "Iteration %d with learning rate of %.06f. Current cycle took %02d minutes %02d seconds. Total time elapsed is %02d:%02d:%02d. Moving avg loss(over %d cycles): %.4f, and Validation loss: %.4f", iter, lr,
        cycleTime_M, cycleTime_S, timeElapsed_H, timeElapsed_M, timeElapsed_S, self.printingFrequency, loss, validation_loss))
    self.timer_lastCall = timeElapsed
  end
end
--
return Debugger
--[[local computer = Mediator
local debugger = Debugger(computer)
local debugger1 = Debugger(computer)
computer:addObj(debugger)
computer:addObj(debugger1)
debugger1:send({key = 'start', value = nil})
for i = 1, 100 do
  debugger1:send({key = 'loss', value = math.random()})
end]]--
