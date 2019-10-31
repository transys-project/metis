#ifndef _ROUTING_TABLE_
#define _ROUTING_TABLE_
#include <fstream>
#include <vector>
#include <map>

typedef std::map<std::pair<int, int>, int> RoutingTable;
typedef std::map<std::pair<int, int>, std::list<int>> RoutingPath;

enum class InsertType
{
    Node = 0,
    Path = 1,
    Unknown = 2
};

class NetworkRoutingTable
{
public:
    NetworkRoutingTable()
    {
        this->insertType = InsertType::Unknown;
    }
    ~NetworkRoutingTable() {}
    void addRoutingforNode(int node, RoutingTable routingTable)
    {
        this->insertType = InsertType::Node;
        this->routingTableList[node] = routingTable;
        this->nodeList.emplace_back(node);
    }

    void addRoutingforPath(std::pair<int, int> &&srcDstPair, std::list<int> &&path)
    {
        this->insertType = InsertType::Path;
        this->routingPath[srcDstPair] = path;
    }

    void transformTabletoPath()
    {
        if (this->insertType == InsertType::Path)
        {
            std::cout << "Routing table is stored by path. No Need For Transfrom." << std::endl;
            return;
        }
        this->routingPath = RoutingPath();
        for (auto iterator_src = this->nodeList.begin(); iterator_src != this->nodeList.end(); iterator_src++)
        {
            for (auto iterator_dst = this->nodeList.begin(); iterator_dst != this->nodeList.end(); iterator_dst++)
            {
                if (*iterator_dst == *iterator_src)
                {
                    continue;
                }
                auto src_dst_pair = std::make_pair(*iterator_src, *iterator_dst);
                this->routingPath[src_dst_pair] = std::list<int>();
                this->routingPath[src_dst_pair].emplace_back(*iterator_src);
                while (this->routingPath[src_dst_pair].back() != *iterator_dst)
                {
                    auto previousHop = this->routingPath[src_dst_pair].back();
                    auto nextHop = this->routingTableList[previousHop][src_dst_pair];
                    this->routingPath[src_dst_pair].emplace_back(nextHop);
                }
            }
        }
    }

    void transformPathToTable()
    {
        if (this->insertType == InsertType::Node)
        {
            std::cout << "Routing table is stored by node. No Need For Transfrom." << std::endl;
            return;
        }
        this->routingTableList = std::map<int, RoutingTable>();
        for (auto node : this->nodeList)
        {
            this->routingTableList[node] = RoutingTable();
        }
        for (auto &pathIterator : this->routingPath)
        {
            auto dst_node = pathIterator.first.second;

            for (auto hopIterator = pathIterator.second.begin(); hopIterator != pathIterator.second.end(); hopIterator++)
            {
                if (*hopIterator == dst_node)
                {
                    continue;
                }
                this->routingTableList[*hopIterator][pathIterator.first] = *(std::next(hopIterator));
            }
        }
    }

    const std::map<int, RoutingTable> &getRoutingTable() const
    {
        return routingTableList;
    }

    const std::vector<int> &getNodeList() const
    {
        return nodeList;
    }

    const RoutingPath &getRoutingPath() const
    {
        return routingPath;
    }

    void serializeRoutingPath(const char *filename) const
    {
        std::ofstream FILE;
        FILE.open(filename);
        FILE << "src,dst,path" << std::endl;
        for (auto &path : this->routingPath)
        {
            FILE << path.first.first << "," << path.first.second << ",[";
            for (auto node : path.second)
            {
                FILE << node;
                if (node != path.first.second)
                {
                    FILE << "|";
                }
            }
            FILE << "]" << std::endl;
        }
        FILE.close();
    }

    void serializeRoutingTable(const char *filename) const
    {
        std::ofstream FILE;
        FILE.open(filename);
        FILE << "node,src,dst,nexthop" << std::endl;
        for (auto &node : this->routingTableList)
        {
            for (auto &routingColumn : node.second)
            {
                FILE << node.first << "," << routingColumn.first.first << "," << routingColumn.first.second << "," << routingColumn.second << std::endl;
            }
        }
        FILE.close();
    }

private:
    std::map<int, RoutingTable> routingTableList;
    std::vector<int> nodeList;
    RoutingPath routingPath;
    InsertType insertType;
};

#endif
