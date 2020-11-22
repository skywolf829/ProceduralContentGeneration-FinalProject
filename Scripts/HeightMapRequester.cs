using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using System;
using System.Threading;

public class HeightMapRequester : RunAbleThread
{
    public static int heightMapSize = 128;

    private bool newRequest = false;
    private bool stop = false;
    private string request;

    public bool hasNewData = false;
    public bool isProcessing = false;
    public bool newNoise = false;
    public bool newStyle = false;
    public bool undo = false;
    public bool canStop = false;
    public bool updateHeightmapResolution = false;
    public int iteration_to_revert_to = 0;
    public int totalIters = 0;
    public int currentIter = 0;
    public bool connected = false;
    public int newStyleForResolution = -1;
    public bool allNewStyles = false;
    public bool updateLockedStyles = false;
    public string lockedStyleRequestString = "";
    public float[,] heightmapvalues = 
        new float[HeightMapRequester.heightMapSize,
        HeightMapRequester.heightMapSize];

    public void RequestNewHeightMap(string request){
        this.request = request;
        newRequest = true;
        isProcessing = true;            
    }
    public void StopServerAndStop(){
        stop = true;
    }
    protected override void Run()
    {
        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");            
            Debug.Log("Connection established");
            connected = true;
            while(Running){
                
                if(stop){
                    stop = false;
                    Debug.Log("Telling server to stop");
                    client.SendFrame("stop");
                    string message = "";
                    bool gotMessage = client.TryReceiveFrameString(out message);
                    this.Stop();
                }
                if(newRequest){
                    string message = "";
                    string iterMessage = "";
                    while(!message.Equals("finished")){
                        //client.SendFrame("requesting_heightmap,"+this.request);
                        client.SendFrame("requesting_heightmap_with_masks,"+this.request);
                        newRequest = false;
                        bool gotMessage = false;
                        bool gotIter = false;
                        double startTime = DateTime.Now.Ticks / 1e7;
                        while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                        {
                            gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                        }
                        if (gotMessage && !message.Equals("finished")){                            
                            byte[] data = System.Convert.FromBase64String(message);
                            heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                            for(int i = 0; i < data.Length; i+=3){
                                int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                                int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                                heightmapvalues[r, c] = (float)data[i] / 255.0f;
                            }
                            client.SendFrame("iterations");
                            startTime = DateTime.Now.Ticks / 1e7;
                            while (!gotIter && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                            {
                                gotIter = client.TryReceiveFrameString(out iterMessage);
                            }
                            if(gotIter){
                                totalIters = Int32.Parse(iterMessage);
                                currentIter = totalIters - 1;
                            }                        
                            hasNewData = true;
                        }
                        else if(gotMessage && message.Equals("finished")){

                        }
                        else{
                            Debug.Log("Response timed out.");
                            message = "finished";
                        }
                        Thread.Yield();
                    }       
                    isProcessing = false;              
                }
                if(updateLockedStyles){
                    updateLockedStyles = false;
                    client.SendFrame("update_locked_styles,"+lockedStyleRequestString);
                    string message = "";
                    bool gotMessage = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                    }
                    if (gotMessage && message.Equals("finished")){                            
                        
                        Debug.Log("Updated locked styles");
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
                if(newNoise){
                    newNoise = false;
                    client.SendFrame("new_noise");
                    string message = "";
                    string iterMessage = "";
                    bool gotMessage = false;
                    bool gotIter = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                            int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        Debug.Log("Updated new noise map");
                        client.SendFrame("iterations");
                        startTime = DateTime.Now.Ticks / 1e7;
                        while (!gotIter && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                        {
                            gotIter = client.TryReceiveFrameString(out iterMessage);
                        }
                        if(gotIter){
                            totalIters = Int32.Parse(iterMessage);
                            currentIter = totalIters - 1;
                        }                        
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
                if(allNewStyles){
                    allNewStyles = false;
                    client.SendFrame("all_new_styles");
                    string message = "";
                    string iterMessage = "";
                    bool gotMessage = false;
                    bool gotIter = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message);
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                            int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        Debug.Log("Updated all styles");
                        client.SendFrame("iterations");
                        startTime = DateTime.Now.Ticks / 1e7;
                        while (!gotIter && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                        {
                            gotIter = client.TryReceiveFrameString(out iterMessage);
                        }
                        if(gotIter){
                            totalIters = Int32.Parse(iterMessage);
                            currentIter = totalIters - 1;
                        }                        
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
                if(newStyleForResolution > -1){
                    client.SendFrame("new_style_for_resolution,"+newStyleForResolution);
                    string message = "";
                    string iterMessage = "";
                    bool gotMessage = false;
                    bool gotIter = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message);
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                            int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        Debug.Log("Updated new style for " + newStyleForResolution);
                        client.SendFrame("iterations");
                        startTime = DateTime.Now.Ticks / 1e7;
                        while (!gotIter && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                        {
                            gotIter = client.TryReceiveFrameString(out iterMessage);
                        }
                        if(gotIter){
                            totalIters = Int32.Parse(iterMessage);
                            currentIter = totalIters - 1;
                        }                        
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                    newStyleForResolution = -1;
                }
                if(newStyle){
                    newStyle = false;
                    client.SendFrame("new_style");
                    string message = "";
                    string iterMessage = "";
                    bool gotMessage = false;
                    bool gotIter = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message);
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                            int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        Debug.Log("Updated new style");
                        client.SendFrame("iterations");
                        startTime = DateTime.Now.Ticks / 1e7;
                        while (!gotIter && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                        {
                            gotIter = client.TryReceiveFrameString(out iterMessage);
                        }
                        if(gotIter){
                            totalIters = Int32.Parse(iterMessage);
                            currentIter = totalIters - 1;
                        }                        
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
                if(undo){
                    undo = false;
                    string message = "";
                    client.SendFrame("undo,"+iteration_to_revert_to);
                    bool gotMessage = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (heightMapSize);
                            int c = (int) ((int)(i/3)/heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        hasNewData = true;
                        currentIter = iteration_to_revert_to;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
                if(updateHeightmapResolution){
                    updateHeightmapResolution = false;
                    string message = "";
                    client.SendFrame("update_resolution,"+HeightMapRequester.heightMapSize);
                    bool gotMessage = false;
                    double startTime = DateTime.Now.Ticks / 1e7;
                    while (!gotMessage && DateTime.Now.Ticks / 1e7 - startTime < 10.0)
                    {
                        gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                    }
                    if (gotMessage){                            
                        byte[] data = System.Convert.FromBase64String(message);
                        heightmapvalues = 
                            new float[HeightMapRequester.heightMapSize,
                            HeightMapRequester.heightMapSize];
                        for(int i = 0; i < data.Length; i+=3){
                            int r = (int)(i/3) % (HeightMapRequester.heightMapSize);
                            int c = (int) ((int)(i/3)/HeightMapRequester.heightMapSize);
                            heightmapvalues[r, c] = (float)data[i] / 255.0f;
                        }
                        hasNewData = true;
                    }
                    else{
                        Debug.Log("Response timed out");
                        message = "";
                    }
                }
            }
            
        }
        connected = false;
        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet
    }
}