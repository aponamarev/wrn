--[[The mediator pattern is used to reduce coupling between classes that communicate with each other. Instead of classes communicating directly, and thus requiring knowledge of their implementation, the classes send messages via a mediator object.]]--
if not torch then require 'torch' end
if not class then class = require 'class' end

AbstractObject = class('AbstractObject')
function AbstractObject:__init(mediator)
  self.mediator = mediator
end
function AbstractObject:send(msg)
  self.mediator:send(msg, self)
end
function AbstractObject:receive(msg)
  assert(false, "Method should be overriden by a concrete object")
end
local M = class('M')
function M:__init()
  self.obj = {}
end
function M:addObj(object)
  table.insert(self.obj, object)
end
function M:send(msg, obj)
  for _, o in pairs(self.obj) do
    if o ~= obj then
      o:receive(msg)
    end
  end
end
local NewTimer = class('NewTimer', 'AbstractObject')
function NewTimer:__init(name, mediator)
  AbstractObject.__init(self, mediator)
  self.name = name
end
function NewTimer:receive(msg)
  print(string.format("%s received message: %s",self.name, msg))
end
Mediator = M()
--[[local computer = M()
local timer1 = NewTimer("Timer1", computer)
local timer2 = NewTimer("Timer2", computer)
local timer3 = NewTimer("Timer3", computer)
computer:addObj(timer1)
computer:addObj(timer2)
computer:addObj(timer3)
timer1:send("what's up")]]--


