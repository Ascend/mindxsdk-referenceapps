#include "GetConfig.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

vector<string> v={"video_width","video_height","frame_rate","is_singlelane","lane_num","line_s_x",
                  "line_s_y","line_e_x","line_e_y","is_vertical","point_x","point_y","point1_x",
                  "point1_y","point2_x","point2_y","det_threshold","nms_iouthreshold"};
bool isnum(string s)
{
    stringstream sin(s);
    double t;
    char p;
    if(!(sin >> t)){
        return false;
    }
    if(sin >> p){
        return false;
    }
    else{
        return true;
    }
}

bool IsSpace(char c)
{
    if (' ' == c || '\t' == c)
        return true;
    return false;
}
 
bool IsCommentChar(char c)
{
    if (c == COMMENT_CHAR){
        return true;
    }else{
        return false;
    }
}
 
void Trim(string & str)
{
    if (str.empty()) {
        return;
    }
    uint32_t i;
    int start_pos, end_pos;
    for (i = 0; i < str.size(); ++i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    if (i == str.size()) { // 全部是空白字符串
        str = "";
        return;
    }
    start_pos = i;
    for (i = str.size() - 1; i >= 0; --i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    end_pos = i;
    str = str.substr(start_pos, end_pos - start_pos + 1);
}
 
bool AnalyseLine(const string & line, string & key, string & value)
{
    if (line.empty())
        return false;
    int start_pos = 0, end_pos = line.size() - 1, pos;
    if ((pos = line.find(COMMENT_CHAR)) != -1) {
        if (0 == pos) {  // 行的第一个字符就是注释字符
            return false;
        }
        end_pos = pos - 1;
    }
    string new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分
 
    if ((pos = new_line.find(':')) == -1)
        return false;  // 没有:号

    key = new_line.substr(0, pos);
    value = new_line.substr(pos + 1, end_pos + 1- (pos + 1));
 
    Trim(key);
    if (key.empty()) {
        return false;
    }
    Trim(value);
    return true;
}
 
// 读取数据
bool ReadConfig(const string & filename, map<string, string> & m)
{
    m.clear();
    ifstream infile(filename.c_str());
    if (!infile) {
        cout << "file open error!" << endl;
        return false;
    }
    string line, key, value;
    while (getline(infile, line)) {
        if (AnalyseLine(line, key, value)) {
            if(value.empty()){
                cout << "parameter "<<key<<" is empty!" << endl;
                return false;
            }
            if(!isnum(value)){
                cout << "parameter "<<key<<" is illegal!" << endl;
                return false;
            }
            m[key] = value;
        }
    }
    for(uint32_t i=0;i<v.size();i++){
        if(m.find(v[i])==m.end()){
            cout << "parameter "<<key<<" do not exist!" << endl;
            return false;
        }
    }
    infile.close();
    return true;
}
