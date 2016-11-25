--[[
Class Description

TrainerDirector class encapsulates builder object and provides method to assemble and return a product (trainer):

Related class
1. TrainerBuilder
]]--
if not torch then require 'torch' end
if not class then class = require 'class' end
--local commander = require 'trainer/Commander/Commander'

local TrainerDirector = class('TrainerDirector')

function TrainerDirector:__init(obj2store)
  assert(class.type(obj2store)=="TrainerBuilder", "Error: attempt to initialize with a wrong object type ("..class.type(obj2store)..") instead of TrainerBuilder")
  self.storedObject = obj2store
end
--
function TrainerDirector:assemble()
  self.storedObject:createNetwork()
  self.storedObject:createDataProvider()
  return self.storedObject.trainer
end
--
return TrainerDirector