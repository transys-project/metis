#ifndef _AVERAGE_QUEUE_
#define _AVERAGE_QUEUE_

#include <map>

template <typename key_type, typename value_type, typename key_list_type = std::vector<key_type>>
class AverageQueue
{
public:
    AverageQueue(){};
    AverageQueue(key_list_type keyList, value_type presetValue) : keyList(keyList), presetValue(presetValue)
    {
        this->fresh();
    }
    ~AverageQueue()
    {
    }
    void push(key_type key, const value_type value)
    {
        int previousCount = this->countBoard.back()[key];
        value_type previousAvg = this->averageBoard.back()[key];
        this->averageBoard.back()[key] = (previousAvg * previousCount) / (previousCount + 1) + value / (previousCount + 1);
        this->countBoard.back()[key] = previousCount + 1;
    }
    std::vector<std::map<key_type, value_type>> &&fetch()
    {
        return std::move(this->averageBoard);
    }
    void fresh()
    {
        this->averageBoard.push_back(std::map<key_type, value_type>());
        this->countBoard.push_back(std::map<key_type, int>());

        for (auto key : keyList)
        {
            this->averageBoard.back()[key] = presetValue;
            this->countBoard.back()[key] = 0;
        }
    }

private:
    std::vector<std::map<key_type, value_type>> averageBoard;
    std::vector<std::map<key_type, int>> countBoard;

    key_list_type keyList;
    value_type presetValue;
};

#endif
