//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 1992-2015 Andras Varga
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#ifdef _MSC_VER
#pragma warning(disable : 4786)
#endif

#include <map>
#include <omnetpp.h>
#include "Packet_m.h"
#include "../builder/control_m.h"

typedef std::map<std::pair<int, int>, int> RoutingTable; // destaddr -> gateindex
typedef std::map<int, int> AddressGateTable;
using namespace omnetpp;

enum
{
  INIT = 0,
  WAITING_FOR_ADDRESS_GATE_TABLE = FSM_Steady(1),
  WAITING_FOR_ROUTING_TABLE = FSM_Steady(2),
  WAITING_FOR_START = FSM_Steady(3),
  ROUTING = FSM_Steady(4)
};

/**
 * Demonstrates static routing, utilizing the cTopology class.
 */
class Routing : public cSimpleModule
{
private:
  int myAddress;

  cFSM fsm;
  RoutingTable rtable;
  AddressGateTable agtable;

  simsignal_t dropSignal;
  simsignal_t outputIfSignal;

protected:
  void dropPacket(cMessage *msg);
  virtual void initialize() override;
  virtual void handleMessage(cMessage *msg) override;
};

Define_Module(Routing);

void Routing::initialize()
{
  this->myAddress = getParentModule()->par("address");

  this->dropSignal = registerSignal("drop");
  this->outputIfSignal = registerSignal("outputIf");

  this->fsm.setName("fsm");
  scheduleAt(0, new cMessage());
}

void Routing::dropPacket(cMessage *msg)
{
  Packet *pk = check_and_cast<Packet *>(msg);
  emit(dropSignal, (long)pk->getByteLength());
  delete pk;
}

void Routing::handleMessage(cMessage *msg)
{

  FSM_Switch(fsm)
  {
  case FSM_Exit(INIT):
    delete msg;
    FSM_Goto(fsm, WAITING_FOR_ADDRESS_GATE_TABLE);
    break;
  case FSM_Enter(WAITING_FOR_ADDRESS_GATE_TABLE):
    break;
  case FSM_Exit(WAITING_FOR_ADDRESS_GATE_TABLE):
    if (msg->arrivedOn("controllerPort$i"))
    {
      auto controlPacket = check_and_cast<Control *>(msg);
      std::stringstream ss(controlPacket->getCommand());
      std::string line;
      while (getline(ss, line, '\n'))
      {
        if (line.empty())
          continue;
        EV << "line content" << line << std::endl;
        std::vector<int> tokens = cStringTokenizer(line.c_str()).asIntVector();
        if (tokens.size() != 2)
          throw cRuntimeError("wrong line in module file: 2 items required, line: \"%s\"", line.c_str());
        // EV << "address " << tokens[0] << " match port" << tokens[1] << std::endl;
        this->agtable[tokens[0]] = tokens[1];
      }
      delete controlPacket;

      cMessage *response = new cMessage("ack");
      send(response, "controllerPort$o");

      FSM_Goto(fsm, WAITING_FOR_ROUTING_TABLE);
    }
    else
    {
      this->dropPacket(msg);
    }
    break;
  case FSM_Enter(WAITING_FOR_ROUTING_TABLE):
    break;
  case FSM_Exit(WAITING_FOR_ROUTING_TABLE):
    if (msg->arrivedOn("controllerPort$i"))
    {
      auto controlPacket = check_and_cast<Control *>(msg);
      std::stringstream ss(controlPacket->getCommand());
      std::string line;
      while (getline(ss, line, '\n'))
      {
        if (line.empty())
          continue;
        std::vector<int> tokens = cStringTokenizer(line.c_str()).asIntVector();
        if (tokens.size() != 3)
          throw cRuntimeError("wrong line in module file: 3 items required, line: \"%s\"", line.c_str());
        this->rtable[std::make_pair(tokens[0], tokens[1])] = tokens[2];
      }
      delete controlPacket;

      cMessage *response = new cMessage("ack");
      send(response, "controllerPort$o");

      FSM_Goto(fsm, WAITING_FOR_START);
    }
    else
    {
      this->dropPacket(msg);
    }
    break;
  case FSM_Enter(WAITING_FOR_START):
    break;
  case FSM_Exit(WAITING_FOR_START):
    if (msg->arrivedOn("controllerPort$i"))
    {
      send(msg, "localOut");

      cMessage *response = new cMessage("ack");
      send(response, "controllerPort$o");

      FSM_Goto(fsm, ROUTING);
    }
    else
    {
      this->dropPacket(msg);
    }
  case FSM_Enter(ROUTING):
    break;
  case FSM_Exit(ROUTING):
    if (msg->arrivedOn("controllerPort$i"))
    {
      EV << "receive control info" << std::endl;
      delete msg;
    }
    else
    {
      Packet *pk = check_and_cast<Packet *>(msg);
      int srcAddr = pk->getSrcAddr();
      int destAddr = pk->getDestAddr();

      if (destAddr == myAddress)
      {
        EV << "local delivery of packet " << pk->getName() << endl;
        send(pk, "localOut");
        emit(outputIfSignal, -1); // -1: local
        return;
      }

      RoutingTable::iterator it = rtable.find(std::make_pair(srcAddr, destAddr));
      if (it == rtable.end())
      {
        EV << "address " << destAddr << " unreachable, discarding packet " << pk->getName() << endl;
        emit(dropSignal, (long)pk->getByteLength());
        delete pk;
        return;
      }

      int outGateIndex = this->agtable[(*it).second];
      EV << "forwarding packet " << pk->getName() << " on gate index " << outGateIndex << endl;
      pk->setHopCount(pk->getHopCount() + 1);
      emit(outputIfSignal, outGateIndex);

      send(pk, "out", outGateIndex);
    }
    break;
  };
}
