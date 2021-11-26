#include "GetConfig.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

vector<string> v={"video_width","video_height","frame_rate","is_singlelane","lane_num","line_s_x",
                  "line_s_y","line_e_x","line_e_y","is_vertical","point_x","point_y","point1_x",
                  "point1_y","point2_x","point2_y","det_threshold","nms_iouthreshold"};
vector<string> v1={"is_singlelane","lane_num","is_vertical"};
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
    if (' ' == c || '\t' == c){
        return true;
    }
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
    if (line.empty()){
        return false;
    }
    int start_pos = 0, end_pos = line.size() - 1, pos;
    if ((pos = line.find(COMMENT_CHAR)) != -1) {
        if (0 == pos) {  // 行的第一个字符就是注释字符
            return false;
        }
        end_pos = pos - 1;
    }
    string new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分
 
    if ((pos = new_line.find(':')) == -1){
        return false;  // 没有:号
    }

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
    bool is_null = true;
    while (getline(infile, line)) {
        if (AnalyseLine(line, key, value)) {
            is_null = false;
            // 判断参数是否不为空
            if(value.empty()){
                cout << "parameter "<<key<<" is empty!" << endl;
                return false;
            }
            // 判断value是否为合法数字
            if(!isnum(value)){
                cout << "parameter "<<key<<" is illegal!" << endl;
                return false;
            }
            m[key] = value;
        }
    }
    if(is_null){
        cout << "config file is null!" << endl;
        return false;
    }
    // 判断is_singlelane,lane_num,is_vertical取值的合法性
    for(uint32_t i=0;i<v1.size();i++){
        if(stoi(m[v1[i]])!=0&&stoi(m[v1[i]])!=1){
            cout << "parameter "<<v1[i]<<" value is illegal!" << endl;
            return false;
        }
    }
    // 判断是否读取到每个参数
    for(uint32_t i=0;i<v.size();i++){
        if(m.find(v[i])==m.end()){
            cout << "parameter "<<v[i]<<" do not exist!" << endl;
            return false;
        }
    }
    infile.close();
    return true;
}
