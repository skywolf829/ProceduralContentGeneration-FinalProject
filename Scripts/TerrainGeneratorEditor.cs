using UnityEngine;
using System.Collections;
using UnityEditor;
using Unity.EditorCoroutines.Editor;
using System.Collections.Generic;
using System.Text;

[CustomEditor(typeof(HeightMapGenerator))]
public class TerrainGeneratorEditor : Editor
{
    public static class EditorGUILayoutUtility
    {
        public static readonly Color DEFAULT_COLOR = new Color(0f, 0f, 0f, 0.3f);
        public static readonly Vector2 DEFAULT_LINE_MARGIN = new Vector2(2f, 2f);

        public const float DEFAULT_LINE_HEIGHT = 1f;

        public static void HorizontalLine(Color color, float height, Vector2 margin)
        {
            GUILayout.Space(margin.x);

            EditorGUI.DrawRect(EditorGUILayout.GetControlRect(false, height), color);

            GUILayout.Space(margin.y);
        }
        public static void HorizontalLine(Color color, float height) => EditorGUILayoutUtility.HorizontalLine(color, height, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(Color color, Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(color, DEFAULT_LINE_HEIGHT, margin);
        public static void HorizontalLine(float height, Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, height, margin);

        public static void HorizontalLine(Color color) => EditorGUILayoutUtility.HorizontalLine(color, DEFAULT_LINE_HEIGHT, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(float height) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, height, DEFAULT_LINE_MARGIN);
        public static void HorizontalLine(Vector2 margin) => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, DEFAULT_LINE_HEIGHT, margin);

        public static void HorizontalLine() => EditorGUILayoutUtility.HorizontalLine(DEFAULT_COLOR, DEFAULT_LINE_HEIGHT, DEFAULT_LINE_MARGIN);

    #if UNITY_EDITOR
    #endif
    }
    private int padding = 10;

    private int maxRect = 500;
    private int textureResolution = 128;
    private float brushSize = 10;
    private float brushIntensity = 1.0f;

    private bool connected = false;

    private Rect lastRect = new Rect();    
    private Rect lastBackgroundRect = new Rect();
    private Rect backgroundRect = new Rect();
    private Texture2D lossTexture;

    private List<int> lossSelected = new List<int>();
    private List<bool> lossEnabled = new List<bool>();
    private List<float[,]> lossMasks = new List<float[,]>();
    private List<Color> lossMaskColors = new List<Color>();
    private List<Rect> lossRects = new List<Rect>();
    private List<Rect> relativeLossRects = new List<Rect>();
    private int lossIndSelected = 0;
    float currentWidth = 0;

    private bool holdingRect = false;
    private bool movingRect = false;
    private bool scalingRectx = false;
    private bool scalingRecty = false;
    private Rect selectedRect = new Rect();
    private int rectIndSelected = 0;
    private int rectGUIclicked = 0;

    private Vector2 lastMouseClickPosition = new Vector2();
    private Vector2 lastMouseDragPosition = new Vector2();

    private float lastTextureUpdate = 0;
    private float updateEvery = 0.016f;

    string[] lossOptions = new string[]
    {
        "Exact height", "Raise", "Lower", "Ridges", "Smooth", 
        "Increase horizontal alignment", 
        "Decrease horizontal alignment", "Increase vertical alignment", 
        "Decrease vertical alignment" 
    };
    
    private List<string> masksToRequestStrings(){
        List<string> requestStrings = new List<string>();
        for(int i = 0; i < lossMasks.Count; i++){
            if(lossEnabled[i]){
                StringBuilder sb = new StringBuilder();
                sb.Append(lossOptions[lossSelected[i]]);
                for(int x = 0; x < textureResolution; x++){
                    for(int y = 0; y < textureResolution; y++){
                        sb.Append(","+((int)(lossMasks[i][x,y]*255)));
                    }
                }
                requestStrings.Add(sb.ToString());
            }
        }
        Debug.Log("This takes a long time?");
        return requestStrings;
    }

    private void updateLossTexture(){
        lossTexture = new Texture2D(textureResolution, textureResolution);
        
        for(int r = 0; r < textureResolution; r++){
            for(int c = 0; c < textureResolution; c++){
                Color colorForPixel = new Color(0, 0, 0, 0);
                float rIntensity = 0f;
                float gIntensity = 0f;
                float bIntensity = 0f;
                float aIntensity = 0f;
                
                for(int i = 0; i < lossMasks.Count; i++){
                    if(lossEnabled[i]){
                        rIntensity += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].r;
                        gIntensity += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].g;
                        bIntensity += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].b;
                        aIntensity += lossMasks[i][textureResolution - 1 - r, c] ;
                    }
                }
                for(int i = 0; i < lossMasks.Count; i++){
                    if(lossEnabled[i]){
                        colorForPixel.r += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].r * (lossMaskColors[i].r / rIntensity);
                        colorForPixel.g += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].g * (lossMaskColors[i].g / gIntensity);
                        colorForPixel.b += lossMasks[i][textureResolution - 1 - r, c] *
                        lossMaskColors[i].b * (lossMaskColors[i].b / bIntensity);
                        colorForPixel.a += lossMasks[i][textureResolution - 1 - r, c] * 
                        (lossMasks[i][textureResolution - 1 - r, c] / aIntensity);
                    }
                }
                //colorForPixel = colorForPixel / numEnabled;
                lossTexture.SetPixel(c, r,
                Color.Lerp(colorForPixel, Color.black, 1-colorForPixel.a));         
            }
        }
        lossTexture.Apply();
    }
    private void AddRule(){
        lossEnabled.Add(true);
        float[,] init_mask = new float[textureResolution, textureResolution];
        lossMasks.Add(init_mask);
        lossMaskColors.Add(new Color(1, 1, 1, 1.0f));
        lossSelected.Add(0);
        lossIndSelected = lossMasks.Count - 1;
    }
    private void RemoveRule(int i){
        lossMasks.RemoveAt(i);
        lossEnabled.RemoveAt(i);
        lossMaskColors.RemoveAt(i);
        lossSelected.RemoveAt(i);
    }
    private void updateRelativeRects(){
        for(int i = 0; i < lossRects.Count; i++){
            Rect absoluteRect = lossRects[i];
            Rect relativeRect = relativeLossRects[i];
            relativeRect.x = (absoluteRect.x - backgroundRect.x) / backgroundRect.width;
            relativeRect.y = (absoluteRect.y - backgroundRect.y) / backgroundRect.height;
            relativeRect.width = absoluteRect.width / backgroundRect.width;
            relativeRect.height = absoluteRect.height / backgroundRect.height;
            relativeLossRects[i] = relativeRect;
        }
    }
    private void updateRelativeRectsClamp(){
        for(int i = 0; i < lossRects.Count; i++){
            Rect absoluteRect = lossRects[i];
            Rect relativeRect = relativeLossRects[i];
            
            if(relativeRect.width < 0){
                relativeRect.x += relativeRect.width;
                relativeRect.width *= -1;
            }
            if(relativeRect.height < 0){
                relativeRect.y += relativeRect.height;
                relativeRect.height *= -1;
            }
            if(relativeRect.x < 0){
                relativeRect.width += relativeRect.x;
                relativeRect.x = 0;
            }            
            if(relativeRect.x > 1){
                relativeRect.width -= (relativeRect.x - 1);
                relativeRect.x = 1.0f;
            }
            if(relativeRect.y < 0){
                relativeRect.height += relativeRect.y;
                relativeRect.y = 0;
            }            
            if(relativeRect.y > 1){
                relativeRect.height -= (relativeRect.y - 1);
                relativeRect.y = 1.0f;
            }
            if(relativeRect.x+relativeRect.width > 1){
                relativeRect.width -= (relativeRect.x + relativeRect.width - 1);
            }
            if(relativeRect.y+relativeRect.height > 1){
                relativeRect.height -= (relativeRect.y + relativeRect.height - 1);
            }



            relativeLossRects[i] = relativeRect;
        }
    }
    private void updateAbsoluteRects(){
        for(int i = 0; i < lossRects.Count; i++){
            Rect absoluteRect = lossRects[i];
            Rect relativeRect = relativeLossRects[i];

            absoluteRect.x = backgroundRect.x + backgroundRect.width * relativeRect.x;
            absoluteRect.y = backgroundRect.y + backgroundRect.height * relativeRect.y;
            absoluteRect.width = relativeRect.width * backgroundRect.width;
            absoluteRect.height = relativeRect.height * backgroundRect.height;
            lossRects[i] = absoluteRect;
        }
    }
    private bool clickInRect(Vector2 position, Rect box){
        return position.x > box.x &&
        position.x < box.x + box.width &&
        position.y > box.y &&
        position.y < box.y + box.height;
    }
    private void mouseDownEvent(){
        Vector2 mousePos = Event.current.mousePosition;
        lastMouseClickPosition = mousePos;
        if(clickInRect(mousePos, backgroundRect)){
            Vector2 relativeMousePos = new Vector2();
            relativeMousePos.x = Mathf.Clamp((mousePos.x - backgroundRect.x) 
            / backgroundRect.width, 0.0f, 1.0f);
            relativeMousePos.y = Mathf.Clamp((mousePos.y - backgroundRect.y)
            / backgroundRect.height, 0.0f, 1.0f);
            drawCircle(relativeMousePos);
        }
        /*
        if(clickInRect(mousePos, backgroundRect)){
            // Check if the box we have selected is clicked first
            if(!holdingRect && rectGUIclicked < lossRects.Count){
                if(clickInRect(mousePos, lossRects[rectGUIclicked])){
                    holdingRect = true;
                    selectedRect = lossRects[rectGUIclicked];
                    rectIndSelected = rectGUIclicked;
                    

                    movingRect = true;
                    lastMouseDragPosition = mousePos;
                }
            }
            // Then check all others
            for(int i = 0; i < lossRects.Count && !holdingRect; i++){
                if(clickInRect(mousePos, lossRects[i])){
                    holdingRect = true;
                    selectedRect = lossRects[i];
                    rectIndSelected = i;
                    rectGUIclicked = i;

                    movingRect = true;
                    lastMouseDragPosition = mousePos;
                }
            }

            // If no existing box was clicked, make a new one
            if(!holdingRect){
                Rect newRect = new Rect(mousePos.x, mousePos.y, 1f, 1f);
                lastMouseDragPosition = mousePos;
                holdingRect = true;
                scalingRectx = true;
                scalingRecty = true;
                selectedRect = newRect;
                lossRects.Add(newRect);
                lossEnabled.Add(true);
                lossMasks.Add(new bool[textureResolution, textureResolution]);
                lossMaskColors.Add(new Color(1, 1, 1, 1));
                relativeLossRects.Add(new Rect(mousePos.x, mousePos.y, 0.001f, 0.001f));
                lossSelected.Add(0);
                rectGUIclicked = lossRects.Count - 1;
                rectIndSelected = lossRects.Count - 1;
            }
        }
        else{
            
        }
        */
    }

    private void drawCircle(Vector2 relativeMousePos){
        int absMousePosx = (int)(relativeMousePos.x *(textureResolution));
        int absMousePosy = (int)(relativeMousePos.y *(textureResolution));
        if(lossIndSelected >= 0 && lossIndSelected < lossMasks.Count
        && lossEnabled[lossIndSelected]){
            int x_start = (int)(Mathf.Clamp(absMousePosx - brushSize / 2f, 0, textureResolution));
            int x_end = (int)(Mathf.Clamp(x_start + brushSize, 0, textureResolution));
            int y_start = (int)(Mathf.Clamp(absMousePosy - brushSize / 2f, 0, textureResolution));
            int y_end = (int)(Mathf.Clamp(y_start + brushSize, 0, textureResolution));
            for(int x = x_start; x < x_end; x++){
                for(int y = y_start; y < y_end; y++){
                    float dist = 0;
                    float xdist = x - absMousePosx;
                    float ydist = y - absMousePosy;
                    dist = Mathf.Sqrt(xdist*xdist + ydist*ydist);
                    if(dist < brushSize / 2f){
                        lossMasks[lossIndSelected][y, x] = brushIntensity;
                    }
                }    
            }
        }
    }
    private void mouseDragEvent(){
        Vector2 mousePos = Event.current.mousePosition;
        if(clickInRect(mousePos, backgroundRect)){
            Vector2 relativeMousePos = new Vector2();
            relativeMousePos.x = Mathf.Clamp((mousePos.x - backgroundRect.x) 
            / backgroundRect.width, 0.0f, 1.0f);
            relativeMousePos.y = Mathf.Clamp((mousePos.y - backgroundRect.y)
            / backgroundRect.height, 0.0f, 1.0f);
            drawCircle(relativeMousePos);
        }
        /*
        if(holdingRect){
            Rect aRect = lossRects[rectIndSelected];
            
            if(scalingRectx){
                float diff = (mousePos.x - lastMouseDragPosition.x);
                //Debug.Log("Scaling in x: " + diff);
                aRect.width = aRect.width + diff;                
            } 
            
            if(scalingRecty){
                float diff = (mousePos.y - lastMouseDragPosition.y);
                //Debug.Log("Scaling in y: " + diff);
                aRect.height = aRect.height + diff;                    
            }
            if(movingRect){
                float diffx = (mousePos.x - lastMouseDragPosition.x);
                float diffy = (mousePos.y - lastMouseDragPosition.y);
                aRect.x = aRect.x + diffx;       
                aRect.y = aRect.y + diffy;      
            }
            lossRects[rectIndSelected] = aRect;
            updateRelativeRects();
        }
        lastMouseDragPosition = mousePos;
        */
    }
    private void mouseUpEvent(){
        /*
        holdingRect = false;
        scalingRectx = false;
        scalingRecty = false;
        movingRect = false;
        updateRelativeRectsClamp();
        updateAbsoluteRects();
        updateRelativeRects();
        */
    }
    public override bool RequiresConstantRepaint()
    {
        return true;
    }
    private GUIStyle AreaStyleNoMargin { 
        get { 
            GUIStyle s = new GUIStyle(EditorStyles. textArea) 
            { 
                margin = new RectOffset(0, 0, 0, 0), 
            }; 
            return s; 
        }         
    }
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        if (Event.current.type == EventType.MouseMove){
            Repaint();
        }
        else if(Event.current.type == EventType.MouseDown){
            mouseDownEvent();
        }
        else if(Event.current.type == EventType.MouseDrag){
            mouseDragEvent();
        }
        else if(Event.current.type == EventType.MouseUp){
            mouseUpEvent();
        }


        HeightMapGenerator myScript = (HeightMapGenerator)target;

        if(GUILayout.Button("Connect"))
        {            
            connected = myScript.Connect();
        }
        if (Event.current.type == EventType.Repaint){
            lastRect = GUILayoutUtility.GetLastRect();
        }
        if(lastRect != null){
            currentWidth = lastRect.width;

            float rectWidth = currentWidth;
            if(rectWidth > maxRect){
                rectWidth = maxRect;
            }
            
            GUILayout.Space(rectWidth + 2*padding);
            float middle = lastRect.x + lastRect.width / 2;
            backgroundRect = new Rect(middle - rectWidth / 2,
            lastRect.y + lastRect.height + padding,
            rectWidth, rectWidth);
            if(backgroundRect.x != lastBackgroundRect.x ||
            backgroundRect.y != lastBackgroundRect.y ||
            backgroundRect.width != lastBackgroundRect.width ||
            backgroundRect.height != lastBackgroundRect.height){
                updateAbsoluteRects();
            }
            EditorGUI.DrawRect(backgroundRect, Color.black);
            if(lossTexture != null){
                EditorGUI.DrawPreviewTexture(backgroundRect, lossTexture);
            }
        }


        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Brush size:");
        brushSize = EditorGUILayout.Slider(brushSize, 1, 50);
        GUILayout.Label("Brush intensity:");
        brushIntensity = EditorGUILayout.Slider(brushIntensity, 0.0f, 1.0f);
        EditorGUILayout.EndHorizontal();

        float wid = Screen.width*(1.0f / 6.0f);

        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Rule number", GUILayout.Width(wid)); 
        GUILayout.Label("Rule enabled", GUILayout.Width(wid)); 
        GUILayout.Label("Rule", GUILayout.Width(wid));
        GUILayout.Label("Rule color", GUILayout.Width(wid));
        GUILayout.Label("", GUILayout.Width(wid));
        GUILayout.Label("", GUILayout.Width(wid));
        EditorGUILayout.EndHorizontal();

        //GUILayout.ExpandWidth (false);
        
        for(int i = 0; i < lossMasks.Count; i++){
            if(lossIndSelected == i){
                GUI.color = Color.yellow;
                GUI.contentColor = Color.yellow;
                GUI.backgroundColor = Color.yellow;
            }
            else{
                GUI.color = Color.white;
                GUI.contentColor = Color.white;
                GUI.backgroundColor = Color.white;
            }
            
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(""+i, GUILayout.Width(wid));
            lossEnabled[i] = EditorGUILayout.Toggle("", lossEnabled[i], GUILayout.Width(wid));
            lossSelected[i] = EditorGUILayout.Popup("", lossSelected[i], 
            lossOptions, GUILayout.Width(wid)); 
            lossMaskColors[i] = EditorGUILayout.ColorField("", lossMaskColors[i], GUILayout.Width(wid));
            if(GUILayout.Button("Select")){
                lossIndSelected = i;
            }
            if(GUILayout.Button("Remove")){
                RemoveRule(i);
                i--;
            }
            GUILayout.Label("", GUILayout.Width(wid));
            EditorGUILayout.EndHorizontal();
        }
        GUI.color = Color.white;
        GUI.contentColor = Color.white;
        GUI.backgroundColor = Color.white;
        if(GUILayout.Button("Add rule")){
            AddRule();
        }
        
        /*
        EditorGUILayout.EndHorizontal();
        GUILayout.ExpandWidth (false);
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Rule number"); 
        GUILayout.Label("Rule");
        GUILayout.Label("");
        GUILayout.Label("");
        EditorGUILayout.EndHorizontal();


        //EditorGUILayoutUtility.HorizontalLine(new Vector2(1f, 1f));
        
        for(int i = 0; i < lossRects.Count; i++){
            if(i == rectGUIclicked){
                EditorGUI.DrawRect(lossRects[i], new Color(1.0f, 1.0f, 0f, 0.5f));
            }
            else{
                EditorGUI.DrawRect(lossRects[i], new Color(1.0f, 0f, 0f, 0.5f));
            }
            EditorGUIUtility.labelWidth = 30;

            Rect r = EditorGUILayout.BeginHorizontal(AreaStyleNoMargin);
            GUILayout.Label("Rule " + (i+1));
            lossSelected[i] = EditorGUILayout.Popup("", lossSelected[i], 
            lossOptions); 
            
            //GUILayout.Label(lossRects[i].ToString());
            if(GUILayout.Button("Select")){
                rectGUIclicked = i;
            }
            if(GUILayout.Button("Delete")){
                lossRects.RemoveAt(i);
                relativeLossRects.RemoveAt(i);
                lossSelected.RemoveAt(i);
                i--;
            }
            EditorGUILayout.EndHorizontal();
        }
        */

        
        EditorGUI.BeginDisabledGroup(!connected);
        EditorGUI.BeginDisabledGroup(myScript.IsInProcess());
        if(GUILayout.Button("Train"))
        {
            //myScript.getRequestMessageFromRects(relativeLossRects, lossSelected, lossOptions);
            
            myScript.setRequestFromMasks(masksToRequestStrings());
            EditorCoroutineUtility.StartCoroutine(myScript.GetHeightMapAsync(), this);
        }
        EditorGUILayoutUtility.HorizontalLine(new Vector2(20f, 20f));
        if(GUILayout.Button("New noise"))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.GetNewNoise(), this);
        }
        if(GUILayout.Button("Update resolution"))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.UpdateHeightmapResolution(), this);
        }
        if(GUILayout.Button("Undo"))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.Undo(), this);
        }
        EditorGUI.EndDisabledGroup();
        if(GUILayout.Button("Disconnect"))
        {
            myScript.Disconnect();
            connected = false;
        }
        EditorGUI.EndDisabledGroup();

        Rect tempRect = GUILayoutUtility.GetLastRect();
        if(myScript.getTexture() != null){
            Rect whereToDraw = new Rect();
            whereToDraw.x = backgroundRect.x;
            whereToDraw.y = tempRect.y + tempRect.height + padding;
            whereToDraw.width = backgroundRect.width;
            whereToDraw.height = backgroundRect.width;
            GUILayout.Space(whereToDraw.height + 2*padding);
            //EditorGUI.DrawRect(whereToDraw, Color.white);
            EditorGUI.DrawPreviewTexture(whereToDraw, myScript.getTexture());
        }
        lastBackgroundRect = backgroundRect;
        if(EditorApplication.timeSinceStartup > lastTextureUpdate + updateEvery){
            updateLossTexture();
            lastTextureUpdate = (float)EditorApplication.timeSinceStartup;
        }
    }
}