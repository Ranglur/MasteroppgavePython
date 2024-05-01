# Start with all groups of order 16
allGroups := AllGroups(16);
Print(Length(allGroups), "\n");

# Filter for non-abelian groups
nonAbelianGroups := Filtered(allGroups, g -> not IsAbelian(g));
Print(Length(nonAbelianGroups), "\n");

# Filter for groups with exactly 2 generators
groups2Generators := Filtered(nonAbelianGroups, g -> Length(MinimalGeneratingSet(g)) = 2);
Print(Length(groups2Generators), "\n");

# Filter for groups with 10 conjugacy classes
groups10ConjugacyClasses := Filtered(groups2Generators, g -> Length(ConjugacyClasses(g)) = 10);
Print(Length(groups10ConjugacyClasses), "\n");





GetElementOrderCounts := function(group)
    local elementOrders, orderCounts, order;

    elementOrders := List(Elements(group), Order);

    orderCounts := rec();
    for order in Set(elementOrders) do
        orderCounts.(String(order)) := Size(Filtered(elementOrders, x -> x = order));
    od;
    return orderCounts;
end;



# Function to find the index (second part of the GAP ID) of each group in groupsWithIrrep4
FindGroupIndexes := function(allGroups, candidateGroups)
    local indexes, group, index;
    indexes := [];
    for group in candidateGroups do
        index := Position(allGroups, group);
        Add(indexes, index); # Add the index to the list of indexes
    od;
    return indexes;
end;


for group in groups10ConjugacyClasses do
    Print("Group: ", IdGroup(group), "\n");
    Print(GetElementOrderCounts(group), "\n\n");
od; 

# Output the indices of the groups in the original allGroups list that match all criteria
groupIndices := FindGroupIndexes(allGroups, groups10ConjugacyClasses);
Print("Indices of groups that match all criteria: ", groupIndices, "\n");

