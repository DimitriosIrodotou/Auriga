import numpy as np


class read_data_catalogue:
    def __init__(self, filename):
        
        self.filename = filename
        f = open(self.filename)
        numdatalines = 0
        
        # count data lines
        for line in f.readlines():
            if line.startswith("#") or not line.strip():
                continue
            numdatalines += 1
        
        print
        'In file', self.filename, 'number of lines with valid data', numdatalines
        
        f.close()
        
        self.associated = np.chararray(numdatalines, itemsize=5)
        
        self.extinction = np.zeros(numdatalines)
        self.dmodulus = np.zeros(numdatalines)
        self.errordmodulusup = np.zeros(numdatalines)
        self.errordmodulusdown = np.zeros(numdatalines)
        
        self.heliodistance = np.zeros(numdatalines)
        self.errorheliodistanceup = np.zeros(numdatalines)
        self.errorheliodistancedown = np.zeros(numdatalines)
        
        self.MWdistance = np.zeros(numdatalines)
        self.MWvelocity = np.zeros(numdatalines)
        
        self.M31distance = np.zeros(numdatalines)
        self.M31velocity = np.zeros(numdatalines)
        
        self.LGdistance = np.zeros(numdatalines)
        self.LGvelocity = np.zeros(numdatalines)
        
        self.Vappmagnitude = np.zeros(numdatalines)
        self.errorVappmagnitudeup = np.zeros(numdatalines)
        self.errorVappmagnitudedown = np.zeros(numdatalines)
        
        self.ellipticity = np.zeros(numdatalines)
        
        self.Vabsmagnitude = np.zeros(numdatalines)
        self.errorVabsmagnitudeup = np.zeros(numdatalines)
        self.errorVabsmagnitudedown = np.zeros(numdatalines)
        
        self.halflightrad = np.zeros(numdatalines)
        self.errorhalflightup = np.zeros(numdatalines)
        self.errorhalflightdown = np.zeros(numdatalines)
        
        self.stellarmass = np.zeros(numdatalines)
        
        self.sigmastars = np.zeros(numdatalines)
        self.errorsigmastarsup = np.zeros(numdatalines)
        self.errorsigmastarsdown = np.zeros(numdatalines)
        
        self.vrot = np.zeros(numdatalines)
        self.errorvrot = np.zeros(numdatalines)
        
        self.HImass = np.zeros(numdatalines)
        self.sigmaHI = np.zeros(numdatalines)
        self.errorsigmaHI = np.zeros(numdatalines)
        self.vrotHI = np.zeros(numdatalines)
        self.errorvrotHI = np.zeros(numdatalines)
        
        self.dynamicalmass = np.zeros(numdatalines)
        
        self.metallicity = np.zeros(numdatalines)
        self.errormetallicity = np.zeros(numdatalines)
    
    def read_table(self):
        f = open(self.filename)
        numdatalines = 0
        
        for line in f.readlines():
            if line.startswith("#") or not line.strip():
                continue
            
            splittedline = line.split()
            
            self.associated[numdatalines] = splittedline[1]
            
            self.extinction[numdatalines] = splittedline[4]
            self.dmodulus[numdatalines] = splittedline[5]
            self.errordmodulusup[numdatalines] = splittedline[6]
            self.errordmodulusdown[numdatalines] = splittedline[7]
            
            self.heliodistance[numdatalines] = splittedline[8]
            self.errorheliodistanceup[numdatalines] = splittedline[9]
            self.errorheliodistancedown[numdatalines] = splittedline[10]
            
            self.MWdistance[numdatalines] = splittedline[11]
            self.MWvelocity[numdatalines] = splittedline[12]
            
            self.M31distance[numdatalines] = splittedline[13]
            self.M31velocity[numdatalines] = splittedline[14]
            
            self.LGdistance[numdatalines] = splittedline[15]
            self.LGvelocity[numdatalines] = splittedline[16]
            
            self.Vappmagnitude[numdatalines] = splittedline[17]
            self.errorVappmagnitudeup[numdatalines] = splittedline[18]
            self.errorVappmagnitudedown[numdatalines] = splittedline[19]
            
            self.ellipticity[numdatalines] = splittedline[20]
            
            self.Vabsmagnitude[numdatalines] = splittedline[21]
            self.errorVabsmagnitudeup[numdatalines] = splittedline[22]
            self.errorVabsmagnitudedown[numdatalines] = splittedline[23]
            
            self.halflightrad[numdatalines] = splittedline[24]
            self.errorhalflightup[numdatalines] = splittedline[25]
            self.errorhalflightdown[numdatalines] = splittedline[26]
            
            self.stellarmass[numdatalines] = splittedline[27]
            
            self.sigmastars[numdatalines] = splittedline[28]
            self.errorsigmastarsup[numdatalines] = splittedline[29]
            self.errorsigmastarsdown[numdatalines] = splittedline[30]
            
            self.vrot[numdatalines] = splittedline[31]
            self.errorvrot[numdatalines] = splittedline[32]
            
            self.HImass[numdatalines] = splittedline[33]
            self.sigmaHI[numdatalines] = splittedline[34]
            self.errorsigmaHI[numdatalines] = splittedline[35]
            self.vrotHI[numdatalines] = splittedline[36]
            self.errorvrotHI[numdatalines] = splittedline[37]
            
            self.dynamicalmass[numdatalines] = splittedline[38]
            
            self.metallicity[numdatalines] = splittedline[39]
            self.errormetallicity[numdatalines] = splittedline[40]
            
            numdatalines += 1
        
        print
        'Read', numdatalines, 'valid data lines from file', self.filename
        
        f.close()
        
        self.transform_to_solar_masses()
        self.average_half_light_radii()
    
    def transform_to_solar_masses(self):
        self.stellarmass *= 1.0e6
        self.HImass *= 1.0e6
        self.dynamicalmass *= 1.0e6
    
    def average_half_light_radii(self):
        self.halflightrad = self.halflightrad * np.sqrt(1.0 - self.ellipticity)