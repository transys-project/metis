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

#include <vector>
#include <omnetpp.h>
#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>

#include "Packet_m.h"
#include "../builder/control_m.h"
#include "../utils/average_queue.hpp"
#include "../utils/average_diff_queue.hpp"
using namespace omnetpp;

enum
{
    INIT = 0,
    WAITING_FOR_TRAFFIC_MATRIX = FSM_Transient(1),
    SENDING_PACKAGE = FSM_Steady(1)
};

/**
 * Generates traffic for the network.
 */
class App : public cSimpleModule
{
private:
    // configuration
    int myAddress;
    simtime_t measureInterval;
    simtime_t lastMeasureCycle;
    std::vector<double> trafficDensity;

    double totalTrafficDensity;
    cPar *packetLengthBytes;

    // state
    cMessage *generatePacket;
    long pkCounter;

    // boost-based random generation
    int seed;
    boost::random::mt19937 rng;
    boost::random::exponential_distribution<double> randomWaitTime;
    boost::random::discrete_distribution<int, double> randomTarget;
    cFSM fsm;

    // statistics
    AverageQueue<int, double> endToEndLatency;
    AverageDiffQueue<int, double> endToEndJitter;

public:
    App();
    virtual ~App();

protected:
    virtual void initialize() override;
    virtual void finish() override;
    virtual void handleMessage(cMessage *msg) override;
    double getWaitTime();
    int getTarget();
};

Define_Module(App);

App::App()
{
    generatePacket = nullptr;
}

App::~App()
{
    cancelAndDelete(generatePacket);
}

double App::getWaitTime()
{
    return this->randomWaitTime(this->rng);
}

int App::getTarget()
{
    return this->randomTarget(this->rng) + 1;
}

void App::initialize()
{
    myAddress = par("address");
    measureInterval = par("measureInterval");
    packetLengthBytes = &par("packetLength");

    this->seed = static_cast<int>(par("seed").doubleValue());

    pkCounter = 0;

    lastMeasureCycle = 0;
}

void App::finish()
{
    auto result = this->endToEndLatency.fetch();

    for (int i = 0; i < result.size(); i++)
    {
        for (auto &pair : result[i])
        {
            std::stringstream ss;
            ss << "end_to_end_latency_" << i << "_" << pair.first << "_" << this->myAddress;
            recordScalar(ss.str().c_str(), pair.second);
        }
    }

    auto jitterResult = this->endToEndJitter.fetch();

    for (int i = 0; i < result.size(); i++)
    {
        for (auto& pair : result[i])
        {
            std::stringstream ss;
            ss << "end_to_end_jitter_" << i << "_" << pair.first << "_" << this->myAddress;
            recordScalar(ss.str().c_str(), pair.second);
        }
    }
}

void App::handleMessage(cMessage *msg)
{
    FSM_Switch(fsm)
    {
    case FSM_Exit(INIT):
        FSM_Goto(fsm, WAITING_FOR_TRAFFIC_MATRIX);
        break;
    case FSM_Exit(WAITING_FOR_TRAFFIC_MATRIX):
        if (true)
        {
            auto controlPacket = check_and_cast<Control *>(msg);
            EV << "node " << this->myAddress << " receiving command " << controlPacket->getCommand() << std::endl;
            std::vector<double> tokens = cStringTokenizer(controlPacket->getCommand()).asDoubleVector();

            this->totalTrafficDensity = 0;
            int tokensNumber = 0;
            for (auto trafficDensity : tokens)
            {
                this->trafficDensity.push_back(trafficDensity);
                this->totalTrafficDensity += trafficDensity;
                tokensNumber += 1;
            }
            EV << "node " << this->myAddress << " total sending traffic density " << this->totalTrafficDensity << " " << std::endl;
            EV << "node " << this->myAddress << " seed is " << this->seed << std::endl;
            EV << "node " << this->myAddress << " number of total receiving tokens are " << tokensNumber << std::endl;
            this->rng.seed(this->seed);

            this->randomWaitTime = boost::random::exponential_distribution<double>((1e9 / 1.2e4) * (this->totalTrafficDensity / (this->trafficDensity.size() - 1)));
            std::for_each(this->trafficDensity.begin(), this->trafficDensity.end(), [this](double &n) { n /= this->totalTrafficDensity; });
            this->randomTarget = boost::random::discrete_distribution<int, double>(this->trafficDensity);
            delete controlPacket;

            generatePacket = new cMessage("nextPacket");
            scheduleAt(simTime() + this->getWaitTime(), generatePacket);

            std::vector<int> l(this->trafficDensity.size());
            std::iota(l.begin(), l.end(), 1);

            this->endToEndLatency = AverageQueue<int, double>(l, 0.0);
            this->endToEndJitter = AverageDiffQueue<int, double>(l, 0.0);

            FSM_Goto(fsm, SENDING_PACKAGE);
            break;
        }
    case FSM_Enter(SENDING_PACKAGE):
        break;
    case FSM_Exit(SENDING_PACKAGE):
        if (msg == generatePacket)
        {
            // Sending packet
            int destAddress = this->getTarget();

            char pkname[40];
            sprintf(pkname, "pk-%d-to-%d-#%ld", myAddress, destAddress, pkCounter++);

            Packet *pk = new Packet(pkname);
            pk->setByteLength(this->packetLengthBytes->intValue());
            pk->setKind(intuniform(0, 7));
            pk->setSrcAddr(myAddress);
            pk->setDestAddr(destAddress);
            send(pk, "out");

            scheduleAt(simTime() + this->getWaitTime(), generatePacket);
        }
        else
        {
            // Handle incoming packet
            Packet *pk = check_and_cast<Packet *>(msg);

            if (simTime() > this->lastMeasureCycle + this->measureInterval)
            {
                this->endToEndLatency.fresh();
                this->endToEndJitter.fresh();
                this->lastMeasureCycle = this->lastMeasureCycle + this->measureInterval;
                this->endToEndLatency.push(pk->getSrcAddr(), static_cast<const double>(SIMTIME_DBL(simTime() - pk->getCreationTime())));
                this->endToEndJitter.push(pk->getSrcAddr(), static_cast<const double>(SIMTIME_DBL(simTime() - pk->getCreationTime())));
            }
            else
            {
                this->endToEndLatency.push(pk->getSrcAddr(), static_cast<const double>(SIMTIME_DBL(simTime() - pk->getCreationTime())));
                this->endToEndJitter.push(pk->getSrcAddr(), static_cast<const double>(SIMTIME_DBL(simTime() - pk->getCreationTime())));
            }

            delete pk;
        }
        break;
    }
}
