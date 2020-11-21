using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Loss : MonoBehaviour
{
    public string GetRequestString(){
        int heightmapsize = HeightMapRequester.heightMapSize;
        float width = transform.localScale.x * 10;
        float height = transform.localScale.z * 10;
        float x1 = ((transform.localPosition.x - (width / 2f)) + 5) / 10f;
        float x2 = ((transform.localPosition.x + (width / 2f)) + 5) / 10f;
        float y1 = ((transform.localPosition.z - (height /2f)) + 5) / 10f;
        float y2 = ((transform.localPosition.z + (height /2f)) + 5) / 10f;
        string requestString = "";
        requestString = requestString + gameObject.tag;        
        requestString = requestString + ",";
        requestString = requestString + (int)(Mathf.Clamp(x1*(heightmapsize-1), 0, (heightmapsize-1)));        
        requestString = requestString + ",";
        requestString = requestString + (int)(Mathf.Clamp(x2*(heightmapsize-1), 0, (heightmapsize-1)));        
        requestString = requestString + ",";
        requestString = requestString + (int)(Mathf.Clamp(y1*(heightmapsize-1), 0, (heightmapsize-1)));        
        requestString = requestString + ",";
        requestString = requestString + (int)(Mathf.Clamp(y2*(heightmapsize-1), 0, (heightmapsize-1)));
        return requestString;
    }
}
