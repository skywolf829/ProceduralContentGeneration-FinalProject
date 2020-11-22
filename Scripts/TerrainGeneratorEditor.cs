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
    private int heightMapRes = 128;
    private int textureResolution = 128;
    private float brushSize = 10;
    private float brushIntensity = 1.0f;

    private bool connected = false;
    private bool[] styleLocked = new bool[6];
    private Rect lastRect = new Rect();    
    private Rect lastBackgroundRect = new Rect();
    private Rect backgroundRect = new Rect();
    private Texture2D lossTexture;

    private List<int> lossSelected = new List<int>();
    private List<bool> lossEnabled = new List<bool>();
    private List<float[,]> lossMasks = new List<float[,]>();
    private List<Color> lossMaskColors = new List<Color>();
    private int lossIndSelected = 0;
    float currentWidth = 0;

    private bool showInstructions = false;
    private bool showTipsAndTricks = false;
    private bool showDebugging = false;
    
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
        return requestStrings;
    }
    private bool atLeastOneStyleUnlocked(){
        bool oneUnlocked = false;
        for(int i = 0; i < styleLocked.Length && !oneUnlocked; i++){
            oneUnlocked = !styleLocked[i];
        }
        return oneUnlocked;
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
    
    private void mouseDownEvent(){
        Vector2 mousePos = Event.current.mousePosition;

        if(clickInRect(mousePos, backgroundRect)){
            Vector2 relativeMousePos = new Vector2();
            relativeMousePos.x = Mathf.Clamp((mousePos.x - backgroundRect.x) 
            / backgroundRect.width, 0.0f, 1.0f);
            relativeMousePos.y = Mathf.Clamp((mousePos.y - backgroundRect.y)
            / backgroundRect.height, 0.0f, 1.0f);
            drawCircle(relativeMousePos);
        }
    }

    private bool clickInRect(Vector2 pos, Rect r){
        return pos.x > r.x && pos.x < r.x+r.width &&
        pos.y > r.y && pos.y < r.y+r.height;
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
        
    }
    private void mouseUpEvent(){
        
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

        EditorGUILayoutUtility.HorizontalLine(new Vector2(20f, 20f));
        EditorStyles.label.wordWrap = true;
        showInstructions = EditorGUILayout.Foldout(showInstructions, "Instructions");
        if(showInstructions){
            EditorGUILayout.LabelField("This tool is connects Unity to a Python-based server " +
            "that hosts a neural network trained to generate realistic looking heightmaps.");
            EditorGUILayout.LabelField("The output heightmaps are resolved at 128x128, but trained with 512x512 " +
            " heightmaps with a 3 arcsecond " +
            "resolution, which means that each pixel is 90m by 90m. Therefore, the returned heightmaps " +
            "are of size 46km x 46km and will have features of that size, making it not a great " +
            "tool if you aim to make a smaller terrain");
            EditorGUILayout.LabelField("First, ensure the python server is running. With 'python server_for_unity.py'");
            EditorGUILayout.LabelField("Next, click the 'connect' button in this tool. When you'd like to shut "+
            "the server down, click disconnect (required, otherwise the server will continue to run " +
            "after closing Unity)");
            EditorGUILayout.LabelField("The heightmap generation process uses two main components: "+
            "the style and the noise. The style guides the overall look of the current heightmap and "+
            "dictates key features like mountain ranges and canyons. The noise gives small perturbations " +
            "to the output from a specific style. Clicking 'new noise' and 'new style' will " +
            "give you a new noise or style to work with.");
            EditorGUILayout.LabelField("Every time a change is made, it is saved and can be reverted to by "+
            "using the 'iteration' slider.");
            GUILayout.Space(10);
            EditorGUILayout.LabelField("The main control over the neural network comes in the form of rules.");
            EditorGUILayout.LabelField("Rules allow the user to tell the neural network what kinds of output " +
            "we want. There rules are as follows:");
            EditorGUILayout.LabelField("Exact height: the network will try to match the exact height given to it");
            EditorGUILayout.LabelField("Raise: the network will try to maximize height in certain areas");
            EditorGUILayout.LabelField("Lower: the network will try to minimize height in certain areas");
            EditorGUILayout.LabelField("Ridges: the network will try to make this area more slanted/rigid");
            EditorGUILayout.LabelField("Smooth: the network will try to smooth out this area");
            EditorGUILayout.LabelField("Increase horizontal alignment: the network will try to make patterns "+
            "that follow a left-right pattern");
            EditorGUILayout.LabelField("Decrease horizontal alignment: the network will try to avoid patterns "+
            "that follow a left-right pattern");
            EditorGUILayout.LabelField("Increase vertical alignment: the network will try to make patterns "+
            "that follow an up-down pattern");
            EditorGUILayout.LabelField("Decrease vertical alignment: the network will try to make patterns "+
            "that follow an up-down pattern");
            EditorGUILayout.LabelField("These rules can be added by clicking the 'add rule' button, and then " +
            "selecting the rule desired with the dropdown. Then, the user can draw in the space above.");
            EditorGUILayout.LabelField("Drawing will draw with a brush diameter chosen with the brush size slider, and an "+
            "intensity of the intensity slider. The intensity determines how important it is for the "+
            "network to follow the rule in that location. High intensity (near 1) means we really want "+
            "the network to have this rule at this location, and low intensity means that it is not as " +
            "important. When intensity is 0 (default across the entire rule), the network won't try to "+
            "follow the rule at all in that spot. The only time intensity is something else is when Exact height is used, in " +
            "which case the network will try to match the exact heightmap drawn.");
            EditorGUILayout.LabelField("When drawing, only the rule selected (highlighted yellow) is updated. "+
            "A rule can be selected by clicking 'Select' next to it.");
            EditorGUILayout.LabelField("The color for different rules can be changed with the rule color.");
            EditorGUILayout.LabelField("Rules can be enabled or disabled by clicking the checkmark for " +
            "rule enabled.");
            
            EditorGUILayout.LabelField("When a desired ruleset is made, the user can click 'Train' to begin " +
            "the optimization process. The network will try to create a realistic heightmap " + 
            "that follows the rules created. While it is training for 100 steps, the terrain " +
            "will be updated in realtime. Any iteration during training can be reverted to " +
            "by chosing the iteration using the slider.");
            }

        
        
        GUILayout.Space(10);
        showTipsAndTricks = EditorGUILayout.Foldout(showTipsAndTricks, "Tips and tricks");
        if(showTipsAndTricks){
            EditorGUILayout.LabelField("1. Having too steep changes too close can make it difficult sometimes " +
            "for the network to optimize. Consider adding a buffer zone of 0 intensity between " + 
            "large changes. For instance, if you'd like high elevation on the top and low elevation " +
            "on the bottom half, perhaps leave the middle third of the rule area with 0 intensity " +
            "so the network can be flexible in that area.");        
            EditorGUILayout.LabelField("2. If the trained output heightmap doesn't match the rules close " +
            "enough for your liking you can click 'train' again.");
            EditorGUILayout.LabelField("3. If the output is close to what you like, try clicking " + 
            "new noise to see small changes in your output");        
            EditorGUILayout.LabelField("4. Use colormaps to give easy visualizations for rules you created. " +
            "For instance, use green for a rule to increase horizontal orientation, and red " +
            "to reduce horizontal orientation.");        
            EditorGUILayout.LabelField("5. Avoid setting the resolution above 128. It will slow down performance.");
        }
        GUILayout.Space(10);
        showDebugging = EditorGUILayout.Foldout(showDebugging, "Debugging");
        if(showDebugging){
            EditorGUILayout.LabelField("If the server seems unresponsive, check the python script. " +
            "If it is still alive, try clicking connect again in Unity and try your command again. "+
            "If that still doesn't work, you'll have to click connect, disconnect, and then start over again.");
        }

        EditorGUILayoutUtility.HorizontalLine(new Vector2(20f, 20f));

        HeightMapGenerator myScript = (HeightMapGenerator)target;

        EditorGUILayout.BeginHorizontal();
        if(GUILayout.Button("Connect"))
        {            
            connected = myScript.Connect();
        }
        
        if(GUILayout.Button("Disconnect"))
        {
            myScript.Disconnect();
            connected = false;
        }
        EditorGUILayout.EndHorizontal();
        
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
        
        EditorGUI.BeginDisabledGroup(!connected);
        EditorGUI.BeginDisabledGroup(myScript.IsInProcess());
        if(GUILayout.Button("Train"))
        {
            if(atLeastOneStyleUnlocked()){
                myScript.setRequestFromMasks(masksToRequestStrings());
                EditorCoroutineUtility.StartCoroutine(myScript.GetHeightMapAsync(), this);
            }
            else{
                Debug.Log("Need at least one style unlocked to train");
            }
        }
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Iteration:");
        int tempIterationChosen = EditorGUILayout.IntSlider(
            myScript.GetCurrentIteration(), 0, myScript.GetTotalIterations()-1);
        if(tempIterationChosen != myScript.GetCurrentIteration()){
            EditorCoroutineUtility.StartCoroutine(myScript.SetCurrentIter(tempIterationChosen), this);
        }        
        EditorGUILayout.EndHorizontal();

        EditorGUILayoutUtility.HorizontalLine(new Vector2(20f, 20f));
        
        wid = Screen.width*(1f/3f);

        EditorGUILayout.BeginHorizontal();

        if(GUILayout.Button("New noise", GUILayout.Width(wid)))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.GetNewNoise(), this);
        }
        if(GUILayout.Button("All new styles", GUILayout.Width(wid)))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.GetAllNewStyles(), this);
        }
        /*
        if(GUILayout.Button("New style"))
        {
            EditorCoroutineUtility.StartCoroutine(myScript.GetNewStyle(), this);
        }
        */
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Style size", GUILayout.Width(wid));
        GUILayout.Label("Get new random style", GUILayout.Width(wid));
        GUILayout.Label("Lock in training", GUILayout.Width(wid));
        EditorGUILayout.EndHorizontal();

        for(int i = 0; i < 6; i++){
            EditorGUILayout.BeginHorizontal();
            int currSize = (int)(Mathf.Pow(2, 2+i));
            GUILayout.Label(currSize+"x"+currSize+ " style", GUILayout.Width(wid));
            if(GUILayout.Button("New style", GUILayout.Width(wid)))
            {
                EditorCoroutineUtility.StartCoroutine(myScript.GetNewStyle(i), this);
            }
            bool tempBool = EditorGUILayout.Toggle("", styleLocked[i], GUILayout.Width(wid));
            if(tempBool != styleLocked[i]){
                styleLocked[i] = tempBool;
                EditorCoroutineUtility.StartCoroutine(myScript.UpdateLockedStyles(styleLocked), this);
            }
            EditorGUILayout.EndHorizontal();
        }
        
        EditorGUILayout.BeginHorizontal();
        int tempHeightMapRes = EditorGUILayout.IntSlider(heightMapRes, 32, 1024); 
        int exp = (int)(Mathf.Log(tempHeightMapRes, 2));
        heightMapRes = (int)(Mathf.Pow(2, exp));
        if(GUILayout.Button("Update resolution"))
        {
            myScript.SetHeightMapResolution(heightMapRes);
            EditorCoroutineUtility.StartCoroutine(myScript.UpdateHeightmapResolution(), this);
        }
        EditorGUILayout.EndHorizontal();
        
        EditorGUI.EndDisabledGroup();
        EditorGUI.EndDisabledGroup();

        Rect tempRect = GUILayoutUtility.GetLastRect();
        if(myScript.getTexture() != null){
            Rect whereToDraw = new Rect();
            whereToDraw.x = backgroundRect.x;
            whereToDraw.y = tempRect.y + tempRect.height + padding;
            whereToDraw.width = backgroundRect.width;
            whereToDraw.height = backgroundRect.width;
            GUILayout.Space(whereToDraw.height + 2*padding);
            EditorGUI.DrawPreviewTexture(whereToDraw, myScript.getTexture());
        }
        lastBackgroundRect = backgroundRect;
        if(EditorApplication.timeSinceStartup > lastTextureUpdate + updateEvery){
            updateLossTexture();
            lastTextureUpdate = (float)EditorApplication.timeSinceStartup;
        }
    }
}