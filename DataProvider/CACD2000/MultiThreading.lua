if not torch then require 'torch' end
if not class then class = require 'class' end
if not Threads then Threads = require 'threads' end
local MultiThreading = class('MultiThreading')
function MultiThreading:__init(f, n_threads, n_jobs, env_setup, variable_setup)
  self.n_jobs = n_jobs
  self.f = f
  --[[
When deserializing a callback, upvalues must be of known types. Since f1,f2,... in threads.Threads() are deserialized in order, we suggest that you make a separated f1 containing all the definitions and put the other code in f2,f3,.... e.g.
  
require 'nn'
local threads = require 'threads'
local model = nn.Linear(5, 10)
threads.Threads(
    2,
    function(idx)                       -- This code will crash
        require 'nn'                    -- because the upvalue 'model' 
        local myModel = model:clone()   -- is of unknown type before deserialization
    end
)

require 'nn'
local threads = require 'threads'
local model = nn.Linear(5, 10)
threads.Threads(
    2,
    function(idx)                      -- This code is OK.
        require 'nn'
    end,                               -- child threads know nn.Linear when deserializing f2
    function(idx)
        local myModel = model:clone()  -- because f1 has already been executed
    end
)
]]--
  Threads.Threads.serialization('threads.sharedserialize')
  local numberOfThreads = n_threads or 5
  torch.setnumthreads(numberOfThreads)
  self.threads = Threads(numberOfThreads, env_setup, variable_setup)
  self.results = {}
  self.endcallback = function(callback)
    local idx, result
    if type(callback) == 'table' then
      idx, result = table.unpack(callback)
    else
      result = callback
    end
    if result then
      self.results[#self.results+1] = result
    end
    if self.threads:hasjob() then
      self.threads:dojob()
    end
    collectgarbage()
  end
end
function MultiThreading:launch()
  for i = 1, self.n_jobs do
    self.threads:addjob(self.f, self.endcallback)
  end
end
function MultiThreading:get_set()
  self.threads:synchronize()
  collectgarbage()
  return self.results
end
--Two actions:
--1. Return data minibatch
--2. Check if you reached the end of the minibatch set
function MultiThreading:get()
  assert(self.n_jobs and self.n_jobs>=2, "Error: MultiThreading requires to provide the total number of jobs (n_jobs) greater than 1. Only "..self.n_jobs .. " were provided.")
  --1. initialize a set counter if the minibatch set is not initialized yet (lazy instantiation)
  if not self.__job_tracker or not self.__extracted_set then
    self.__job_tracker = 0
    self:launch()
    self.__extracted_set = self:get_set()
  end
  --2. If the last call delivered the first minibatch out of the set (setCounter = 1), launch the data retrival process again
  if self.__job_tracker == 1 then
    self:launch() --no need to check opt.nBatches
  end
  --3. if you reached the end of the set counter get a new set and reset the counter
  if self.__job_tracker == self.n_jobs then
    self.__extracted_set = self:get_set()
    self.__job_tracker = 0
  end
--4. increment the counter to track where you are in the minibatchcontainer
  self.__job_tracker = self.__job_tracker + 1
--5. return one minibatch
  return table.remove(self.__extracted_set)
end
return MultiThreading
--[[local MiniBatch = require 'MiniBatch'
local opt = {
  descriptor = '/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/cacd.t7',
  dataset_path = '/Users/aponamaryov/Downloads/CACD2000',
  scale = 256,
  flip = {vertircal = nil},
  crop = nil,
  fivecrop = 224,
  batch_size = 50
}
local mini = MiniBatch(opt)


local multithread = MultiThreading(function()
    print("Theread " .. __threadid .. " was successfully executed.")
    for zz = 1, 4 do 
      print(math.random(1,20))
    end
    return {__threadid, thread_mini:get()}
  end, 4, 4,
  function()
    if not torch then require 'torch' end
    MiniBatch = require 'MiniBatch'
  end,
  function()
    math.randomseed(__threadid)
    thread_mini = mini
  end)
local t = torch.Timer()
for i = 1, 8 do
  batch = multithread:get()
  print("Batch was obtained in " .. t:time().real .. " seconds.")
  print("Request " .. i .. ": ")
  print(batch.data:float():size())
  image.display(batch.data)
end]]--
