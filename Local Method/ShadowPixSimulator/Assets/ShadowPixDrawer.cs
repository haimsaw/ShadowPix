using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;

public class ShadowPixDrawer : MonoBehaviour
{

    public string ReceiversPath;
    public string X_castersPath;
    public string Y_castersPath;

    public GameObject lightA;
    public GameObject lightB;
    public GameObject lightC;
    public GameObject lightD;

    void Start()
    {
        float[][] receivers = ReadMatrix(ReceiversPath);
        float[][] x_casters = ReadMatrix(X_castersPath);
        float[][] y_casters = ReadMatrix(Y_castersPath);

        float receiver_size_x = 1f;
        float receiver_size_y = 1f;
        float x_caster_size_x = 0.001f;
        float x_caster_size_y = receiver_size_y;
        float y_caster_size_x = receiver_size_x + x_caster_size_x;
        float y_caster_size_y = x_caster_size_x;

        // bug in last x_caster
        for (int iRow = 0; iRow < receivers.Length; iRow++)
        {
            for (int iCol = 0; iCol < receivers[0].Length; iCol++)
            {
                GameObject x_caster = GameObject.CreatePrimitive(PrimitiveType.Cube);
                GameObject y_caster = GameObject.CreatePrimitive(PrimitiveType.Cube);
                GameObject receiver = GameObject.CreatePrimitive(PrimitiveType.Cube);

                x_caster.GetComponent<Renderer>().material.color = new Color(255, 255, 255);
                y_caster.GetComponent<Renderer>().material.color = new Color(255, 255, 255);
                receiver.GetComponent<Renderer>().material.color = new Color(255, 255, 255);

                x_caster.transform.localScale = new Vector3(x_caster_size_x, x_caster_size_y, x_casters[iRow][iCol]);
                y_caster.transform.localScale = new Vector3(y_caster_size_x, y_caster_size_y, y_casters[iRow][iCol]);
                receiver.transform.localScale = new Vector3(receiver_size_x, receiver_size_y, receivers[iRow][iCol]);

                x_caster.transform.position = new Vector3((-receiver_size_x / 2 - x_caster_size_x / 2) + iCol * (receiver_size_x + x_caster_size_x), iRow * (receiver_size_y + y_caster_size_y), x_casters[iRow][iCol] / 2.0f);
                y_caster.transform.position = new Vector3(-x_caster_size_x / 2 + iCol * y_caster_size_x, receiver_size_y / 2 + y_caster_size_y / 2 + iRow * (receiver_size_y + y_caster_size_y), y_casters[iRow][iCol] / 2.0f);
                receiver.transform.position = new Vector3(iCol * (receiver_size_x + x_caster_size_x), iRow * (receiver_size_y + y_caster_size_y), receivers[iRow][iCol] / 2.0f);
            }
        }

    }

    private float[][] ReadMatrix(string filePath)
    {
        StreamReader reader = new StreamReader(filePath);
        string file_content = reader.ReadToEnd();
        file_content = file_content.Replace("[", "");
        file_content = file_content.Replace("]", "");

        var cleanedRows = Regex.Split(file_content, @"}\s*,\s*{")
                    .Select(r => r.Replace("{", "").Replace("}", "").Trim())
                    .ToList();

        var matrix = new float[cleanedRows.Count][];
        for (var i = 0; i < cleanedRows.Count; i++)
        {
            var data = cleanedRows.ElementAt(i).Split(',');
            matrix[i] = data.Select(c => float.Parse(c.Trim())).ToArray();
        }

        return matrix;
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void onToggleAClick(bool value)
    {
        lightA.SetActive(value);
    }

    public void onToggleBClick(bool value)
    {
        lightB.SetActive(value);
    }

    public void onToggleCClick(bool value)
    {
        lightC.SetActive(value);
    }

    public void onToggleDClick(bool value)
    {
        lightD.SetActive(value);
    }
}
