//==========================================================================
//  NETBUILDER.CC - part of
//
//                     OMNeT++/OMNEST
//            Discrete System Simulation in C++
//
//==========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 1992-2015 Andras Varga

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <omnetpp.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/random_spanning_tree.hpp>
#include <boost/graph/property_iter_range.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include "control_m.h"
#include "../utils/routing_table.hpp"
#include "../utils/traffic_matrix.hpp"

using namespace boost;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef std::vector<std::pair<std::pair<int, int>, int>> RouterTable;

enum
{
    INIT = 0,
    BUILD_NETWORK = FSM_Transient(1),
    DISTRIBUTE_ADDRESS_GATE_MAP = FSM_Transient(2),
    WAITING_FOR_RESPONSE_GATE_MAP = FSM_Steady(1),
    DISTRIBUTE_ROUTING = FSM_Transient(3),
    WAITING_FOR_RESPONSE_ROUTING = FSM_Steady(2),
    DISTRIBUTE_TRAFFIC_MATRIX = FSM_Transient(4),
    WAITING_FOR_RESPONSE_TRAFFIC = FSM_Steady(3),
    FINISH = FSM_Steady(4)
};

using namespace omnetpp;

Control *generateControlPacket(const char *name, std::string command)
{
    Control *controlPacket = new Control(name);
    controlPacket->setCommand(command.c_str());
    return controlPacket;
}

/**
 * Builds a network dynamically, with the topology coming from a
 * text file.
 */
class NetBuilder : public cSimpleModule
{
protected:
    void connect(cGate *src, cGate *dest, double delay, double ber, double datarate);
    void buildNetwork(cModule *parent);
    void generateRandomTrafficMatrix(int trafficFactor);
    void sendAddressGateMap();
    void sendRoutingTable();
    void sendTrafficMatrix();
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;

private:
    cFSM fsm;
    Graph *topo;

    int seed;
    int trafficFactor;

    boost::random::mt19937 rng;

    std::map<long, cModule *> nodeid2mod;
    std::map<long, std::map<int, int>> addressGateTableList;
    NetworkRoutingTable routingTable;
    TrafficMatrix trafficMatrix;
    int receiving_count;
};

Define_Module(NetBuilder);

void NetBuilder::initialize()
{
    // build the network in event 1, because it is undefined whether the simkernel
    // will implicitly initialize modules created *during* initialization, or this needs
    // to be done manually.
    scheduleAt(0, new cMessage());
    topo = nullptr;
    this->fsm.setName("builder_fsm");
    this->seed = par("seed").intValue();
    this->trafficFactor = par("trafficFactor").intValue();

    this->rng.seed(this->seed);
    this->trafficMatrix = TrafficMatrix(this->seed);

    EV << "random seed of NetBuilder is " << this->seed << std::endl;
    EV << "traffic factor of NetBuilder is " << this->trafficFactor << std::endl;
}

void NetBuilder::handleMessage(cMessage *msg)
{
    FSM_Switch(fsm)
    {
    case FSM_Exit(INIT):
        FSM_Goto(fsm, BUILD_NETWORK);
        break;

    case FSM_Exit(BUILD_NETWORK):
        if (!msg->isSelfMessage())
            throw cRuntimeError("This module does not process messages.");
        buildNetwork(getParentModule());
        FSM_Goto(fsm, DISTRIBUTE_ADDRESS_GATE_MAP);
        break;

    case FSM_Exit(DISTRIBUTE_ADDRESS_GATE_MAP):
        this->sendAddressGateMap();
        this->receiving_count = 0;
        delete msg;
        FSM_Goto(fsm, WAITING_FOR_RESPONSE_GATE_MAP);
        break;

    case FSM_Enter(WAITING_FOR_RESPONSE_GATE_MAP):
        break;

    case FSM_Exit(WAITING_FOR_RESPONSE_GATE_MAP):
        this->receiving_count += 1;
        delete msg;
        if (this->receiving_count == this->nodeid2mod.size())
        {
            FSM_Goto(fsm, DISTRIBUTE_ROUTING);
        }
        break;
    case FSM_Exit(DISTRIBUTE_ROUTING):
        this->sendRoutingTable();
        this->receiving_count = 0;
        FSM_Goto(fsm, WAITING_FOR_RESPONSE_ROUTING);
        break;
    case FSM_Enter(WAITING_FOR_RESPONSE_ROUTING):
        break;
    case FSM_Exit(WAITING_FOR_RESPONSE_ROUTING):
        this->receiving_count += 1;
        delete msg;
        if (this->receiving_count == this->nodeid2mod.size())
        {
            this->receiving_count = 0;
            FSM_Goto(fsm, DISTRIBUTE_TRAFFIC_MATRIX);
        }
    case FSM_Exit(DISTRIBUTE_TRAFFIC_MATRIX):
        this->trafficMatrix.generateRandomTrafficMatrix(this->nodeid2mod.size(), this->trafficFactor);
        EV << "generate traffix matrix with size " << this->nodeid2mod.size() << " " << this->nodeid2mod.size() << std::endl;
        this->trafficMatrix.serializeTrafficMatrix("./results/traffix_matrix.txt");
        this->sendTrafficMatrix();
        FSM_Goto(fsm, WAITING_FOR_RESPONSE_TRAFFIC);
        break;
    case FSM_Enter(WAITING_FOR_RESPONSE_TRAFFIC):
        break;
    case FSM_Exit(WAITING_FOR_RESPONSE_TRAFFIC):
        this->receiving_count += 1;
        delete msg;
        if (this->receiving_count == this->nodeid2mod.size())
        {
            FSM_Goto(fsm, FINISH);
        }
        break;
    case FSM_Enter(FINISH):
        break;
    case FSM_Exit(FINISH):
        delete msg;
        break;
    };
}

void NetBuilder::connect(cGate *src, cGate *dest, double delay, double ber, double datarate)
{
    cDatarateChannel *channel = nullptr;
    if (delay > 0 || ber > 0 || datarate > 0)
    {
        channel = cDatarateChannel::create("channel");
        if (delay > 0)
            channel->setDelay(delay);
        if (ber > 0)
            channel->setBitErrorRate(ber);
        if (datarate > 0)
            channel->setDatarate(datarate);
    }
    src->connectTo(dest, channel);
}

void NetBuilder::sendAddressGateMap()
{
    int gate = 0;
    for (auto addressGateTable : addressGateTableList)
    {
        std::stringstream ss;
        for (auto addressGateColumn : addressGateTable.second)
        {
            ss << addressGateColumn.first << " " << addressGateColumn.second << std::endl;
        }

        auto addressGateMessage = generateControlPacket("address-gate map", ss.str());
        send(addressGateMessage, "out", gate);
        gate++;
    }
}

void NetBuilder::sendRoutingTable()
{

    this->routingTable.transformPathToTable();
    this->routingTable.serializeRoutingPath("./results/routing_path.txt");
    this->routingTable.serializeRoutingTable("./results/routing_table.txt");

    const auto &routerTableList = this->routingTable.getRoutingTable();
    int gate = 0;
    for (auto routertable : routerTableList)
    {
        std::stringstream ss;
        for (auto routerColumn : routertable.second)
        {
            ss << routerColumn.first.first << " " << routerColumn.first.second << " " << routerColumn.second << std::endl;
        }
        auto routerMessage = generateControlPacket("router table", ss.str());
        send(routerMessage, "out", gate);
        gate++;
    }
}

void NetBuilder::sendTrafficMatrix()
{
    int gate = 0;
    for (auto trafficColumn : this->trafficMatrix.getTrafficMatrix())
    {
        std::stringstream ss;
        for (auto trafficDensity : trafficColumn)
        {
            ss << trafficDensity << " ";
        }
        auto trafficMessage = generateControlPacket("traffic matrix", ss.str());
        send(trafficMessage, "out", gate);
        gate++;
    }
}

void NetBuilder::buildNetwork(cModule *parent)
{

    std::string line;
    std::fstream nodesFile(par("nodesFile").stringValue(), std::ios::in);
    while (getline(nodesFile, line, '\n'))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::vector<std::string> tokens = cStringTokenizer(line.c_str()).asVector();
        if (tokens.size() != 3)
            throw cRuntimeError("wrong line in module file: 3 items required, line: \"%s\"", line.c_str());

        // get fields from tokens
        long nodeid = atol(tokens[0].c_str());
        const char *name = tokens[1].c_str();
        const char *modtypename = tokens[2].c_str();
        EV << "NODE id=" << nodeid << " name=" << name << " type=" << modtypename << "\n";

        // create module
        cModuleType *modtype = cModuleType::find(modtypename);
        if (!modtype)
            throw cRuntimeError("module type `%s' for node `%s' not found", modtypename, name);
        cModule *mod = modtype->create(name, parent);
        nodeid2mod[nodeid] = mod;

        // read params from the ini file, etc
        mod->finalizeParameters();
    }

    topo = new Graph(nodeid2mod.size());

    for (auto &pair : nodeid2mod)
    {
        addressGateTableList[pair.first] = std::map<int, int>();
    }

    // read and create connections
    std::fstream connectionsFile(par("connectionsFile").stringValue(), std::ios::in);
    while (getline(connectionsFile, line, '\n'))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::vector<std::string> tokens = cStringTokenizer(line.c_str()).asVector();
        if (tokens.size() != 5)
            throw cRuntimeError("wrong line in parameters file: 5 items required, line: \"%s\"", line.c_str());

        // get fields from tokens
        long srcnodeid = atol(tokens[0].c_str());
        long destnodeid = atol(tokens[1].c_str());
        double delay = tokens[2] != "-" ? atof(tokens[2].c_str()) : -1;
        double error = tokens[3] != "-" ? atof(tokens[3].c_str()) : -1;
        double datarate = tokens[4] != "-" ? atof(tokens[4].c_str()) : -1;

        if (nodeid2mod.find(srcnodeid) == nodeid2mod.end())
            throw cRuntimeError("wrong line in connections file: node with id=%ld not found", srcnodeid);
        if (nodeid2mod.find(destnodeid) == nodeid2mod.end())
            throw cRuntimeError("wrong line in connections file: node with id=%ld not found", destnodeid);

        // add edge in boost graph
        // IMPORTANT: boost mark node from 0
        add_edge(srcnodeid - 1, destnodeid - 1, *topo);

        int srcNodeGateSize = static_cast<int>(addressGateTableList[srcnodeid].size());
        int dstNodeGateSize = static_cast<int>(addressGateTableList[destnodeid].size());
        addressGateTableList[srcnodeid][destnodeid] = srcNodeGateSize;
        addressGateTableList[destnodeid][srcnodeid] = dstNodeGateSize;

        cModule *srcmod = nodeid2mod[srcnodeid];
        cModule *destmod = nodeid2mod[destnodeid];

        cGate *srcIn, *srcOut, *destIn, *destOut;
        srcmod->getOrCreateFirstUnconnectedGatePair("port", false, true, srcIn, srcOut);
        destmod->getOrCreateFirstUnconnectedGatePair("port", false, true, destIn, destOut);

        // connect
        connect(srcOut, destIn, delay, error, datarate);
        connect(destOut, srcIn, delay, error, datarate);
    }

    // connect builder with node.
    // TODO: Maybe separate builder and controller.
    for (auto it = nodeid2mod.begin(); it != nodeid2mod.end(); ++it)
    {
        cGate *controllerIn, *controllerOut;
        cGate *nodeIn, *nodeOut;
        controllerIn = this->getOrCreateFirstUnconnectedGate("in", 0, false, true);
        controllerOut = this->getOrCreateFirstUnconnectedGate("out", 0, false, true);
        nodeIn = it->second->gate("controllerPort$i");
        nodeOut = it->second->gate("controllerPort$o");

        connect(controllerOut, nodeIn, 0.01, 0, 1e6);
        connect(nodeOut, controllerIn, 0.01, 0, 1e6);
    }

    std::map<long, cModule *>::iterator it;

    // final touches: buildinside, initialize()
    for (it = nodeid2mod.begin(); it != nodeid2mod.end(); ++it)
    {
        cModule *mod = it->second;
        mod->buildInside();
    }

    // multi-stage init
    bool more = true;
    for (int stage = 0; more; stage++)
    {
        more = false;
        for (it = nodeid2mod.begin(); it != nodeid2mod.end(); ++it)
        {
            cModule *mod = it->second;
            if (mod->callInitialize(stage))
                more = true;
        }
    }

    typedef property_map<Graph, vertex_index_t>::type IndexMap;
    IndexMap topo_index = get(vertex_index, *topo);

#ifdef DEBUG
    for (auto vp = vertices(*topo); vp.first != vp.second; ++vp.first)
    {
        auto v = *vp.first;
        EV << "adjacent vertices of " << topo_index[v] << " is: ";
        typename graph_traits<Graph>::adjacency_iterator ai;
        typename graph_traits<Graph>::adjacency_iterator ai_end;
        for (boost::tie(ai, ai_end) = adjacent_vertices(v, *topo);
             ai != ai_end; ++ai)
            EV << topo_index[*ai] << " ";
        EV << std::endl;
    }
#endif

    for (auto vp = vertices(*topo); vp.first != vp.second; ++vp.first)
    {
        // destination node
        auto dst = *vp.first;

        std::vector<graph_traits<Graph>::vertex_descriptor> p(num_vertices(*topo));
        auto predmap = boost::make_iterator_property_map(p.begin(), topo_index);

        random_spanning_tree(*topo, rng, root_vertex(dst).predecessor_map(predmap));

        for (auto vp_inner = vertices(*topo); vp_inner.first != vp_inner.second; ++vp_inner.first)
        {
            auto src = *vp_inner.first;
            auto src_dst_pair = std::make_pair(topo_index[src] + 1, topo_index[dst] + 1);

            if (src == dst)
            { // no routing table needed.
                continue;
            }

            std::list<int> path;
            path.emplace_back(src + 1);

            if (predmap[topo_index[src]] == topo_index[dst])
            {
                path.emplace_back(topo_index[dst] + 1);
            }
            else
            {
                auto predecessor = topo_index[src];
                while (predecessor != topo_index[dst])
                {
                    path.emplace_back(predmap[predecessor] + 1);
                    predecessor = predmap[predecessor];
                }
            }
            routingTable.addRoutingforPath(std::move(src_dst_pair), std::move(path));
        }
    }
}
