using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Rotate : MonoBehaviour {

    private float sensitivity = 1;
    public Toggle cameraMovementToggle;

    void Start()
    {
        
    }

    void Update()
    {
        if (cameraMovementToggle.isOn)
        {
            var c = this.transform;
            c.Rotate(0, Input.GetAxis("Mouse X") * sensitivity, 0);
            c.Rotate(-Input.GetAxis("Mouse Y") * sensitivity, 0, 0);
            //c.Rotate(0, 0, -Input.GetAxis("QandE") * 90 * Time.deltaTime);
            //if (Input.GetMouseButtonDown(0))
            //    Cursor.lockState = CursorLockMode.Locked;
        }
    }

}
