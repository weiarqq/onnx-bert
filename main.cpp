
#include <iostream>

#include <string>
// #include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>



using namespace std;
using namespace Ort;

void printModelInfo(Ort::Session &session, Ort::AllocatorWithDefaultOptions &allocator)
{
    //输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    cout<<"Number of input node is:"<<num_input_nodes<<endl;
    cout<<"Number of output node is:"<<num_output_nodes<<endl;

    //获取输入输出维度
    for(auto i = 0; i<num_input_nodes;i++)
    {
        std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout<<endl<<"input "<<i<<" dim is: ";
        for(auto j=0; j<input_dims.size();j++)
            cout<<input_dims[j]<<" ";
    }
    for(auto i = 0; i<num_output_nodes;i++)
    {
        std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout<<endl<<"output "<<i<<" dim is: ";
        for(auto j=0; j<output_dims.size();j++)
            cout<<output_dims[j]<<" ";
    }
    //输入输出的节点名
    cout<<endl;//换行输出
    for(auto i = 0; i<num_input_nodes;i++)
        cout<<"The input op-name "<<i<<" is:"<<session.GetInputName(i, allocator)<<endl;
    for(auto i = 0; i<num_output_nodes;i++)
        cout<<"The output op-name "<<i<<" is:"<<session.GetOutputName(i, allocator)<<endl;

    //input_dims_2[0] = input_dims_1[0] = output_dims[0] = 1;//batch size = 1
}

int main(int argc, char const *argv[])
{
    string model_path = "/Users/wangqi/workspace/cpp-workspace/onnx-bert/onnx/model.onnx";

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "PoseEstimate");
    Ort::SessionOptions session_options;
    //CUDA加速开启
    // OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0); //tensorRT
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::AllocatorWithDefaultOptions allocator;
    //加载ONNX模型
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //打印模型的信息
    printModelInfo(session,allocator);
    return 0;
}
