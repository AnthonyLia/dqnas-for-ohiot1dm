import xml.etree.ElementTree as ET

def get_glucose_train_data(numbers):
    if type(numbers) == list:
        pass
    else:
        if numbers in [559,563,570,575,588,591]:
            tree = ET.parse('../OhioT1DM/2018/train/%s-ws-training.xml'%str(numbers))
            #tree = ET.parse('c:/users/a/documents/school/asu/2023 - fall session c/eee 499/OhioT1DM/2018/train/%s-ws-training.xml'%str(numbers))
            #tree = ET.parse('c:/users/A/documents/EEE 499/OhioT1DM/2018/train/%s-ws-training.xml'%str(numbers))
        elif numbers in [540,544,552,567,584,596]:
            tree = ET.parse('../OhioT1DM/2020/train/%s-ws-training.xml'%str(numbers))
            #tree = ET.parse('c:/users/a/documents/school/asu/2023 - fall session c/eee 499/OhioT1DM/2020/train/%s-ws-training.xml'%str(numbers))
            #tree = ET.parse('c:/users/A/documents/EEE 499/OhioT1DM/2020/train/%s-ws-training.xml'%str(numbers))
        else:
            print("Number given is not in the Ohio data, check the number given")

    root = tree.getroot()

    patient_list = []
    patient_glucose_level = []
    patient_finger_stick = []
    patient_basal = []
    patient_temp_basal = []
    patient_bolus = []
    patient_meal = []
    patient_sleep = []
    patient_work = []
    patient_stressors = []
    patient_hypo_event = []
    patient_illness = []
    patient_exercise = []
    patient_basis_heart_rate = []
    patient_basis_gsr = []
    patient_basis_skin_temperature = []
    patient_basis_air_temperature = []
    patient_basis_steps = []
    patient_basis_sleep = []


    for j in range(0,len(root[0])):
        patient_glucose_level.append(list([root[0][j].attrib['ts'],float(root[0][j].attrib['value'])]))
    '''for j in range(0,len(root[1])):
        patient_finger_stick.append(root[1][j].attrib)
    for j in range(0,len(root[2])):
        patient_basal.append(root[2][j].attrib)
    for j in range(0,len(root[3])):
        patient_temp_basal.append(root[3][j].attrib)
    for j in range(0,len(root[4])):
        patient_bolus.append(root[4][j].attrib)
    for j in range(0,len(root[5])):
        patient_meal.append(root[5][j].attrib)
    for j in range(0,len(root[6])):
        patient_sleep.append(root[6][j].attrib)
    for j in range(0,len(root[7])):
        patient_work.append(root[7][j].attrib)
    for j in range(0,len(root[8])):
        patient_stressors.append(root[8][j].attrib)
    for j in range(0,len(root[9])):
        patient_hypo_event.append(root[9][j].attrib)
    for j in range(0,len(root[10])):
        patient_illness.append(root[10][j].attrib)
    for j in range(0,len(root[11])):
        patient_exercise.append(root[11][j].attrib)
    for j in range(0,len(root[12])):
        patient_basis_heart_rate.append(root[12][j].attrib)   
    for j in range(0,len(root[13])):
        patient_basis_gsr.append(root[13][j].attrib)
    for j in range(0,len(root[14])):
        patient_basis_skin_temperature.append(root[14][j].attrib)
    for j in range(0,len(root[15])):
        patient_basis_air_temperature.append(root[15][j].attrib)
    for j in range(0,len(root[16])):
        patient_basis_steps.append(root[16][j].attrib)
    for j in range(0,len(root[17])):
        patient_basis_sleep.append(root[17][j].attrib)
    '''

    return patient_glucose_level

def get_glucose_test_data(numbers):
    if type(numbers) == list:
        pass
    else:
        if numbers in [559,563,570,575,588,591]:
            tree = ET.parse('../OhioT1DM/2018/test/%s-ws-testing.xml'%str(numbers))
            #tree = ET.parse('c:/users/a/documents/school/asu/2023 - fall session c/eee 499/OhioT1DM/2018/test/%s-ws-testing.xml'%str(numbers))
            #tree = ET.parse('c:/users/A/documents/EEE 499/OhioT1DM/2018/test/%s-ws-testing.xml'%str(numbers))
        elif numbers in [540,544,552,567,584,596]:
            tree = ET.parse('../OhioT1DM/2020/test/%s-ws-testing.xml'%str(numbers))
            #tree = ET.parse('c:/users/a/documents/school/asu/2023 - fall session c/eee 499/OhioT1DM/2020/test/%s-ws-testing.xml'%str(numbers))
            #tree = ET.parse('c:/users/A/documents/EEE 499/OhioT1DM/2020/test/%s-ws-testing.xml'%str(numbers))
        else:
            print("Number given is not in the Ohio data, check the number given")

    root = tree.getroot()

    patient_list = []
    patient_glucose_level = []
    patient_finger_stick = []
    patient_basal = []
    patient_temp_basal = []
    patient_bolus = []
    patient_meal = []
    patient_sleep = []
    patient_work = []
    patient_stressors = []
    patient_hypo_event = []
    patient_illness = []
    patient_exercise = []
    patient_basis_heart_rate = []
    patient_basis_gsr = []
    patient_basis_skin_temperature = []
    patient_basis_air_temperature = []
    patient_basis_steps = []
    patient_basis_sleep = []


    for j in range(0,len(root[0])):
        patient_glucose_level.append(list([root[0][j].attrib['ts'],float(root[0][j].attrib['value'])]))
    '''for j in range(0,len(root[1])):
        patient_finger_stick.append(root[1][j].attrib)
    for j in range(0,len(root[2])):
        patient_basal.append(root[2][j].attrib)
    for j in range(0,len(root[3])):
        patient_temp_basal.append(root[3][j].attrib)
    for j in range(0,len(root[4])):
        patient_bolus.append(root[4][j].attrib)
    for j in range(0,len(root[5])):
        patient_meal.append(root[5][j].attrib)
    for j in range(0,len(root[6])):
        patient_sleep.append(root[6][j].attrib)
    for j in range(0,len(root[7])):
        patient_work.append(root[7][j].attrib)
    for j in range(0,len(root[8])):
        patient_stressors.append(root[8][j].attrib)
    for j in range(0,len(root[9])):
        patient_hypo_event.append(root[9][j].attrib)
    for j in range(0,len(root[10])):
        patient_illness.append(root[10][j].attrib)
    for j in range(0,len(root[11])):
        patient_exercise.append(root[11][j].attrib)
    for j in range(0,len(root[12])):
        patient_basis_heart_rate.append(root[12][j].attrib)   
    for j in range(0,len(root[13])):
        patient_basis_gsr.append(root[13][j].attrib)
    for j in range(0,len(root[14])):
        patient_basis_skin_temperature.append(root[14][j].attrib)
    for j in range(0,len(root[15])):
        patient_basis_air_temperature.append(root[15][j].attrib)
    for j in range(0,len(root[16])):
        patient_basis_steps.append(root[16][j].attrib)
    for j in range(0,len(root[17])):
        patient_basis_sleep.append(root[17][j].attrib)
    '''

    return patient_glucose_level
