--[[
Command pattern used to add commands to an object.

The pattern consits of 3 parts:
1. (let's skip this one) Abstraction Protocol to enforce implementation of the common method across all sub-classes
2. Concretion Command to be added to an object
3. Concretion Commander that will hold all the commands

Commander Class
Purpose to agregate the commands train and test in the storage unit
execute these commands through methods train and test
]]--
if not class then class = require 'class' end
if not torch then require 'torch' end
local Test = require 'trainer/Commander/Test'
local Learn = require 'trainer/Commander/Learn'
--local Deploy = require 'trainer/Commander/Deploy'

local Commander = class('Commander')
function Commander:__init(net, opt, dtype)
  self.learner = Learn(net, opt, dtype)
  self.tester = Test(net, opt, dtype)
  self.deployer = nil --Deploy(net, opt)
end
function Commander:learn(data)
  return self.learner:launch(data)
end
function Commander:test(data)
  return self.tester:launch(data)
end
function Commander:deploy(data)
  --return self.deploy:launch(data)
end
return Commander