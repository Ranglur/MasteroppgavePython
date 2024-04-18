# Start with all groups of order 64
allGroups := AllGroups(64);
Print(Length(allGroups),"\n");

# Filter for non-abelian groups
nonAbelianGroups := Filtered(allGroups, g -> not IsAbelian(g));
Print(Length(nonAbelianGroups),"\n");

# Filter for groups with exactly 2 generators
groupsTwoGenerators := Filtered(nonAbelianGroups, g -> Length(MinimalGeneratingSet(g)) = 2);
Print(Length(groupsTwoGenerators),"\n");

# Filter for groups with 22 conjugacy classes
groups22ConjugacyClasses := Filtered(groupsTwoGenerators, g -> Length(ConjugacyClasses(g)) = 22);
Print(Length(groups22ConjugacyClasses),"\n");


# Filter for groups with exponent 8
groupsExponent8 := Filtered(groups22ConjugacyClasses, g -> Exponent(g) = 8);

Print(Length(groupsExponent8),"\n");


# Finally, filter based on the distribution of conjugacy class sizes
groupsCCstructure := Filtered(groupsExponent8, g ->
    (Length(Filtered(ConjugacyClasses(g), c -> Size(c) = 4)) = 12) and
    (Length(Filtered(ConjugacyClasses(g), c -> Size(c) = 2)) = 6)
);

Print(Length(groupsCCstructure),"\n");

groupsCenterC4 := Filtered(groupsCCstructure, g -> 
    IsCyclic(Center(g)) and Size(Center(g)) = 4
);

Print(Length(groupsCenterC4),"\n");


G := groupsCenterC4[1]; # One of your candidate groups
CenterG := Center(G);
DerivedSubgroupG := DerivedSubgroup(G);

# Check if they are of size 4 and coincide
if Size(CenterG) = 4 and Size(DerivedSubgroupG) = 4 and IsSubgroup(CenterG, DerivedSubgroupG) and IsSubgroup(DerivedSubgroupG, CenterG) then
    Print("G has a center equal to its derived subgroup, both of size 4.\n");
fi;

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


for group in groupsCenterC4 do
    Print("Group: ", IdGroup(group), "\n");
    Print(GetElementOrderCounts(group), "\n\n");
od; 

# Output the indices of the groups in the original allGroups list that match all criteria
groupIndices := FindGroupIndexes(allGroups, groupsCenterC4);
Print("Indices of groups that match all criteria: ", groupIndices, "\n");


