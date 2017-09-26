#pragma once

#include "common.hpp"

/// A node of the category tree
class categoryNode
{
public:
	categoryNode(string name, categoryNode* parent, int nodeId);
	~categoryNode();

	void observe(void);
	void addChild(categoryNode* child);

	categoryNode* find(string* category, int L);

	void print(int depth);
	void fprintJson(int depth, FILE* f);

	string name;
	categoryNode* parent;
	map<string, categoryNode*> children;
	int productCount; // How many products belong to this category?
	set<int> productSet;
	int nodeId;
};

/// A complete category hierarchy
class categoryTree
{
public:
	categoryTree(string rootName, bool skipRoot);
	categoryTree();
	~categoryTree();

	void print(void);
	void fprintJson(FILE* f);

	categoryNode* addPath(vector<string> category);

	vector<categoryNode*> pathFromId(int nodeId);
	void incrementCounts(int nodeId);
	int nNodes(void);

	bool skipRoot; // skipRoot should be true if there are multiple "top-level" categories, i.e., if the root node is not a "real" category.
	categoryNode* root;

	map<categoryNode*,int> nodeToId;
	vector<categoryNode*> idToNode;
};