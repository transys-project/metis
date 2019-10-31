#ifndef _AVERAGE_DIFF_QUEUE_H_
#define _AVERAGE_DIFF_QUEUE_H_
#include <map>

template <typename key_type, typename value_type, typename key_list_type = std::vector<key_type>>
class AverageDiffQueue
{
public:
    AverageDiffQueue(){};
    AverageDiffQueue(key_list_type keyList, value_type presetValue) : keyList(keyList), presetValue(presetValue)
    {
        this->fresh();
        for (auto key: keyList) {
            this->previousItemSaved[key] = false;
        }
    }
    ~AverageDiffQueue()
    {
    }
    void push(key_type key, const value_type value)
    {
        if (!this->previousItemSaved[key]) {
            this->previousItem[key] = value;
            return;
        }

        value_type absoluteDiff = (value - this->previousItem[key] > 0)?(value - this->previousItem[key]):(this->previousItem[key] - value);
        this->previousItem[key] = value;

        int previousCount = this->countBoard.back()[key];
        value_type previousAvg = this->averageBoard.back()[key];
        this->averageBoard.back()[key] = (previousAvg * previousCount) / (previousCount + 1) + absoluteDiff / (previousCount + 1);
        this->countBoard.back()[key] = previousCount + 1;
        return;
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
    std::map<key_type, value_type> previousItem;
    std::map<key_type, bool> previousItemSaved;

    key_list_type keyList;
    value_type presetValue;
};

#endif
