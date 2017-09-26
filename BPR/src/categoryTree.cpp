#include "categoryTree.hpp"


categoryNode::categoryNode(	string name, categoryNode* parent, int nodeId) 
							: name(name)
							, parent(parent)
							, productCount(0)
							, nodeId(nodeId)
{
	children = map<string, categoryNode*>();
	productSet = set<int>();
}

categoryNode::~categoryNode()
{
	for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++) {
		delete it->second;
	}
}

/// Should be called every time a product from this category is observed when loading the corpus
void categoryNode::observe(void)
{
	productCount ++;
}

void categoryNode::addChild(categoryNode* child)
{
	children[child->name] = child;
}

/// Walk down a category tree looking for a particular category (list of strings of length L), or return 0 if it doesn't exist
categoryNode* categoryNode::find(string* category, int L)
{
	if (L == 0) {
		return this;
	}
	if (children.find(category[0]) == children.end()) {
		return 0;
	}
	return children[category[0]]->find(category + 1, L - 1);
}

void categoryNode::print(int depth)
{
	for (int d = 0; d < depth; d ++) {
		fprintf(stderr, "  ");
	}
	fprintf(stderr, "%d (%s), count = %d, productset.size = %ld\n", nodeId, name.c_str(), productCount, productSet.size());
	
	for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++) {
		(it->second)->print(depth + 1);
	}
}

/// Print a JSON representation of the category tree to a file
void categoryNode::fprintJson(int depth, FILE* f)
{
	for (int d = 0; d < depth; d ++) {
		fprintf(f, "  ");
	}
	fprintf(f, "{");

	fprintf(f, "\"nodeId\": %d, \"nodeName\": \"%s\", \"observations\": %d", nodeId, name.c_str(), productCount);

	if (children.size() > 0) {
		bool childTopics = false;
		for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++) {
			childTopics = true;
			break;
		}

		if (childTopics) {
			fprintf(f, ", \"children\":\n");
			for (int d = 0; d < depth + 1; d ++) {
				fprintf(f, "  ");
			}
			fprintf(f, "[\n");

			bool first = true;
			for (map<string, categoryNode*>::iterator it = children.begin(); it != children.end(); it ++) {
				if (not first) {
					fprintf(f, ",\n");
				}
				it->second->fprintJson(depth + 2, f);
				first = false;
			}
			fprintf(f, "\n");

			for (int d = 0; d < depth + 1; d ++) {
				fprintf(f, "  ");
			}
			fprintf(f, "]");
		}
	}
	fprintf(f, "\n");

	for (int d = 0; d < depth; d ++) {
		fprintf(f, "  ");
	}
	fprintf(f, "}");
}

categoryTree::categoryTree(string rootName, bool skipRoot) : skipRoot(skipRoot)
{
	root = new categoryNode(rootName, 0, 0);

	nodeToId = map<categoryNode*, int>();
	idToNode = vector<categoryNode*>();

	nodeToId[root] = 0;
	idToNode.push_back(root);
}

categoryTree::categoryTree() : skipRoot(false)
{
	root = new categoryNode("root", 0, 0);

	nodeToId = map<categoryNode*, int>();
	idToNode = vector<categoryNode*>();

	nodeToId[root] = 0;
	idToNode.push_back(root);
}

categoryTree::~categoryTree()
{
	delete root; // Will recursively delete all children
}

void categoryTree::print(void)
{
	root->print(0);
}

void categoryTree::fprintJson(FILE* f)
{
	root->fprintJson(0, f);
}

/// Add a new category to the category tree, whose name is given by a vector of strings
categoryNode* categoryTree::addPath(vector<string> category)
{
	string* categoryP = &(category[0]);
	categoryNode* deepest = root;
	categoryNode* child = 0;
	int L = category.size();

	if (skipRoot) {
		categoryP ++;
		L --;
		if (L == 0) {
			return root;
		}
	}

	while (L > 0 and (child = deepest->find(categoryP, 1))) {
		categoryP ++;
		deepest = child;
		L --;
	}

	// We ran out of children. Need to add the rest.
	while (L > 0) {
		int nextId = (int) idToNode.size();

		child = new categoryNode(categoryP[0], deepest, nextId);
		deepest->addChild(child);
		deepest = child;
		categoryP ++;
		L --;

		// Give each new child a new id. This code should be changed if we only want to give leaf nodes ids.
		nodeToId[child] = nextId;
		idToNode.push_back(child);
	}
	return child;
}

/// From a leaf node (or any node really) get the path of nodes above it going back to the root
vector<categoryNode*> categoryTree::pathFromId(int nodeId)
{
	vector<categoryNode*> pathR;

	categoryNode* current = idToNode[nodeId];
	while (current) {
		pathR.push_back(current);
		current = current->parent;
	}
	reverse(pathR.begin(), pathR.end());
	return pathR;
}

/// Increment all nodes along a path (e.g. for a product in Electronics->Mobile Phones->Accessories increment all three category nodes)
void categoryTree::incrementCounts(int nodeId)
{
	categoryNode* current = idToNode[nodeId];
	while (current) {
		current->observe();
		current = current->parent;
	}
}

int categoryTree::nNodes(void)
{
	return idToNode.size();
}