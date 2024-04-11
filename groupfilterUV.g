# Start with all groups of order 64
allGroups := AllGroups(32);
Print(Length(allGroups), "\n");

# Filter for non-abelian groups
nonAbelianGroups := Filtered(allGroups, g -> not IsAbelian(g));
Print(Length(nonAbelianGroups), "\n");

# Filter for groups with exactly 4 generators
groups4Generators := Filtered(nonAbelianGroups, g -> Length(MinimalGeneratingSet(g)) = 4);
Print(Length(groups4Generators), "\n");

# Filter for groups with 17 conjugacy classes
groups17ConjugacyClasses := Filtered(groups4Generators, g -> Length(ConjugacyClasses(g)) = 17);
Print(Length(groups17ConjugacyClasses), "\n");


# # Finally, filter based on the distribution of conjugacy class sizes
# groupsCCstructure := Filtered(groupsExponent8, g ->
#     (Length(Filtered(ConjugacyClasses(g), c -> Size(c) = 4)) = 12) and
#     (Length(Filtered(ConjugacyClasses(g), c -> Size(c) = 2)) = 6)
# );



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


for group in groupsExponent8 do
    Print("Group: ", IdGroup(group), "\n");
    Print(GetElementOrderCounts(group), "\n\n");
od; 

# Output the indices of the groups in the original allGroups list that match all criteria
groupIndices := FindGroupIndexes(allGroups, groupsExponent8);
Print("Indices of groups that match all criteria: ", groupIndices, "\n");


