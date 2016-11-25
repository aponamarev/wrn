require 'torch'
require 'paths'
local path_to_descriptor = 'wrn/Results'
local path_to_grouped_descriptor = 'wrn/Results/grouped_cacd.t7'

local descriptor = torch.load(paths.concat(path_to_descriptor, 'cacd.t7'))
--
function description_classes(d)
  local class_group = {}
  for i = 1, d.num_classes do
    class_group[i] = {file_name = {}, classes = d.classes[i], age = {}}
  end
  for i = 1,d.size do
    local array_index = #class_group[d.identity_id[i]].file_name + 1
    local class_id = d.identity_id[i]
    table.insert(class_group[class_id].file_name, array_index, d.file_name[i])
    table.insert(class_group[class_id].age, array_index, d.age[i])
  end
  return class_group
end
function train_val_descr(descriptor, numberOfSamples)
  local class_num = #descriptor
  local classes = {}
  --Copy class names into classes table
  for i = 1, class_num do
    classes[i] = descriptor[i].classes
  end
  --Create tables that will contain resulting descriptors for both Val and Train descriptors.
  local val_desc = {
    file_name = {},
    age = {},
    classes = {},
    size = 0,
    identity_id = {},
    num_classes = 0
  }
  local train_desc = { 
    file_name = {},
    age = {},
    classes = {},
    size = 0,
    identity_id = {},
    num_classes = 0
    }
  val_desc.num_classes, train_desc.num_classes = class_num, class_num
  val_desc.classes, train_desc.classes = classes, classes
  for i = 1, class_num do
    local remaining_size = #descriptor[i].file_name
    for ind = 1, numberOfSamples do
      val_desc.size = val_desc.size + 1
      local rand_index = math.random(1, remaining_size+1-ind)
      val_desc.file_name[val_desc.size] = table.remove(descriptor[i].file_name, rand_index)
      val_desc.age[val_desc.size] = table.remove(descriptor[i].age, rand_index)
      val_desc.identity_id[val_desc.size] = i
    end
  end
  for i = 1, class_num do
    local remaining_size = #descriptor[i].file_name
    for ind = 1, remaining_size do
      train_desc.size = train_desc.size + 1
      local rand_index = math.random(1, remaining_size+1-ind)
      train_desc.file_name[train_desc.size] = table.remove(descriptor[i].file_name, rand_index)
      train_desc.age[train_desc.size] = table.remove(descriptor[i].age, rand_index)
      train_desc.identity_id[train_desc.size] = i
    end
  end
  return train_desc, val_desc
end

--
local grouped_descriptor = description_classes(descriptor)
print("grouping is complete")
torch.save(paths.concat(path_to_descriptor, 'grouped_cacd.t7'), grouped_descriptor)
print("grouped file was successfully saved")
local train, val = train_val_descr(grouped_descriptor, 4)
print("train and val descriptors were successfully created")
torch.save(paths.concat(path_to_descriptor, 'cacd_train.t7'), train)
torch.save(paths.concat(path_to_descriptor, 'cacd_val.t7'), val)
--
print("Transformation is complete!")

