using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class HeightMapGenerator : MonoBehaviour
{

    private bool inProcess = false;

    public int currentHeightMapResolution = 128;
    public Terrain terrain;
    //public GameObject HeightmapAttributeEditorParent;

    private HeightMapRequester heightMapRequestor = null;

    private Texture2D heightMapTexture;

    private string currentRequest = "";
    public bool IsInProcess(){
        return inProcess;
    }
    public Texture2D getTexture(){
        return heightMapTexture;
    }
    /*
    public void getRequestMessage(){
        string request = "";
        for(int i = 0; i < HeightmapAttributeEditorParent.transform.childCount; i++){
            if(HeightmapAttributeEditorParent.transform.
            GetChild(i).gameObject.activeSelf){
                if(request.Length > 0){
                    request = request + ",";
                }
                request = request + HeightmapAttributeEditorParent.
                    transform.GetChild(i).gameObject.GetComponent<Loss>().GetRequestString();
            }
        }
        currentRequest = request;
    }*/

    public void getRequestMessageFromRects(List<Rect> rects, List<int> losses, string[] lossStrings){
        string request = "";        
        int heightmapsize = HeightMapRequester.heightMapSize;
        for(int i = 0; i < rects.Count; i++){            
            if(request.Length > 0){
                request = request + ",";
            }
            Rect r = rects[i];
            request = request + lossStrings[losses[i]] + ",";
            request = request + (int)(Mathf.Clamp(r.x*(heightmapsize-1), 0, (heightmapsize-1))) + ",";
            request = request + (int)(Mathf.Clamp((r.x+r.width)*(heightmapsize-1), 0, (heightmapsize-1))) + ",";
            request = request + (int)(Mathf.Clamp(r.y*(heightmapsize-1), 0, (heightmapsize-1))) + ",";
            request = request + (int)(Mathf.Clamp((r.y+r.height)*(heightmapsize-1), 0, (heightmapsize-1)));           
        }
        currentRequest = request;
    }

    public void setRequestFromMasks(List<string> requestStrings){
        string request = "";
        for(int i = 0; i < requestStrings.Count; i++){
            request = request + requestStrings[i];
            if(i < requestStrings.Count - 1){
                request = request + ",";
            }
        }
        currentRequest = request;
    }

    public IEnumerator UpdateHeightmapResolution(){
        if(heightMapRequestor != null && !inProcess){
            bool isfixed = false;
            HeightMapRequester.heightMapSize = currentHeightMapResolution;
            terrain.terrainData.heightmapResolution = currentHeightMapResolution+1;
            terrain.terrainData.size = new Vector3(10, 2, 10);
            heightMapRequestor.updateHeightmapResolution = true;
            inProcess = true;
            while(!isfixed){
                if(heightMapRequestor.hasNewData){
                    heightMapRequestor.hasNewData = false;
                    terrain.terrainData.SetHeights(0, 0, 
                    heightMapRequestor.heightmapvalues);
                    isfixed = true;
                    updateTexture();
                }
                yield return null;
            }
            inProcess = false;
        }
    }

    public bool Connect(){
        Debug.Log("Initializing connection");
        heightMapRequestor = new HeightMapRequester();
        heightMapRequestor.Start();
        return true;
    }

    public void Disconnect(){
        Debug.Log("Ending connection");
        heightMapRequestor.StopServerAndStop();
        heightMapRequestor = null;
        inProcess = false;
    }
    
    private void updateTexture(){        
        int hmw = HeightMapRequester.heightMapSize;
        
        heightMapTexture = new Texture2D(currentHeightMapResolution, 
        currentHeightMapResolution);

        float[,] heights = terrain.terrainData.GetHeights(0, 0, 
        currentHeightMapResolution, currentHeightMapResolution);
        Color[] colors = new Color[currentHeightMapResolution*currentHeightMapResolution];
        for(int i = 0; i < colors.Length; i++){
            int r = i % hmw;
            int c = hmw - 1 - (int)(i / hmw);
            Color col = new Color(heights[r,c],heights[r,c],heights[r,c]);
            colors[i] = col;
        }
        heightMapTexture.SetPixels(colors);
        heightMapTexture.Apply();
    }

    public IEnumerator GetNewNoise(){
        if(heightMapRequestor != null && !inProcess){
            bool isfixed = false;
            heightMapRequestor.newNoise = true;
            inProcess = true;
            
            while(!isfixed){
                if(heightMapRequestor.hasNewData){
                    heightMapRequestor.hasNewData = false;
                    terrain.terrainData.SetHeights(0, 0, 
                    heightMapRequestor.heightmapvalues);
                    isfixed = true;
                    updateTexture();
                }
                yield return null;
            }
            inProcess = false;
        }
    }

    public IEnumerator Undo(){
        if(heightMapRequestor != null && !inProcess){
            bool isfixed = false;
            heightMapRequestor.undo = true;

            inProcess = true;
            while(!isfixed){
                if(heightMapRequestor.hasNewData){
                    heightMapRequestor.hasNewData = false;
                    terrain.terrainData.SetHeights(0, 0, 
                    heightMapRequestor.heightmapvalues);
                    isfixed = true;
                    updateTexture();
                }
                yield return null;
            }
            inProcess = false;
        }
    }
    public IEnumerator GetHeightMapAsync(){
        if(heightMapRequestor != null && !inProcess){
            inProcess = true;
            Debug.Log("Async process started");
            
            while(!heightMapRequestor.connected){
                Debug.Log("Waiting for thread to start");
                yield return null;
            }

            heightMapRequestor.RequestNewHeightMap(currentRequest);

            yield return null;

            while(heightMapRequestor.isProcessing){
                if(heightMapRequestor.hasNewData){
                    heightMapRequestor.hasNewData = false;
                    terrain.terrainData.SetHeights(0, 0, 
                    heightMapRequestor.heightmapvalues);
                }
                yield return null;
            }
            updateTexture();
            inProcess = false;            
        }
        else if (heightMapRequestor == null){
            Debug.Log("You are disconnected");            
        }
        else if(inProcess){
           Debug.Log("You are attempting to make multiple calls at same time");
        }
    }

    private void OnDestroy()
    {        
        heightMapRequestor.StopServerAndStop();
    }
}
