import numpy as np
import cv2
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class image:
    def __init__(self,folder,name,colours,col_ind):
        self.filename = folder + name + '.png'
        #print
        #print name
        img = cv2.imread(self.filename)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        ret, img_th = cv2.threshold(img_gray,127,255,0)
        ret, img_th = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
        
        if colours:
            
            imHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);
            if col_ind < 0:
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
                img_ero = cv2.erode(img_th, kernel) #cv2.MORPH_OPEN, kernel)

                internal_chain, hierarchy = cv2.findContours(img_ero,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
                chain_ar = internal_chain[0]
                HSV = np.zeros(3)
        
                for c in range(chain_ar.shape[0]):
                    pHSV = imHSV[chain_ar[c][0][1]][chain_ar[c][0][0]]; # x, y interchanged
                    if pHSV[0] != 0:
                        HSV = np.vstack((HSV,pHSV))
                        HSV = HSV[1:len(HSV)] 
            
                HSVmu = np.mean(HSV,0)
                in_range = []
                for co in range(len(colours)):
                    in_range.append(sum((HSVmu > colours[co].hsv_min) & (HSVmu < colours[co].hsv_max)))
                    
                col_ind = np.argmax(in_range)
                print(colours[col_ind].name)
                
            HSVmin = np.asarray(colours[col_ind].hsv_min)
            HSVmax = np.asarray(colours[col_ind].hsv_max)
            
            img_th = cv2.inRange(imHSV,HSVmin,HSVmax)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
            img_th = cv2.morphologyEx (img_th, cv2.MORPH_CLOSE, kernel) # erode + dilate


        _, contour_list, self.hierarchy = cv2.findContours(img_th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        self.n_obj = len(contour_list)
        #print self.n_obj
        ch = 0
        while ch <= self.n_obj and len(contour_list[ch]) < 10:
            ch += 1
        self.contour = contour_list[ch]

    
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    x_mu = np.matrix(x - mu)
    determ = np.linalg.det(sigma)
    if determ < 0.01:
        new_diag = np.diag(sigma).copy()
        new_diag[new_diag == 0] = 0.05
        new_sigma = np.matrix(np.diag(new_diag))
        determ = np.linalg.det(new_sigma)
        inv = new_sigma.I
    else:
        inv = sigma.I
         
    norm_const = 1.0/ (math.pow((2*np.pi),float(size)/2) * math.pow(determ,1.0/2))
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result
        

class contour:
    curvature_threshold = 0.08
    k = 4
    polygon_tolerance = 0.04

    def __init__(self,input_contour,features):
        chain = input_contour
        curvature_chain = []
        cont_ar = np.asarray(chain)
    
        # compute axes feature
        ellipse = cv2.fitEllipse(cont_ar)
        (center,axes,orientation) = ellipse
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)
        axes_ratio = minoraxis_length/majoraxis_length
        
        area = cv2.contourArea(cont_ar)
        perimeter = cv2.arcLength(cont_ar,True)
        area_ratio = perimeter / area
        perimeter_ratio = minoraxis_length / perimeter 
        
        epsilon = self.polygon_tolerance*perimeter
        vertex_approx = 1.0 / len(cv2.approxPolyDP(cont_ar,epsilon,True))
        length = len(input_contour)
        
        # compute curvature and convexity features
        for i in range(cont_ar.shape[0]-self.k):
            num = cont_ar[i][0][1]-cont_ar[i-self.k][0][1] # y
            den = cont_ar[i][0][0]-cont_ar[i-self.k][0][0] # x
            angle_prev = -math.atan2(num,den)/math.pi
        
            num = cont_ar[i+self.k][0][1]-cont_ar[i][0][1] # y
            den = cont_ar[i+self.k][0][0]-cont_ar[i][0][0] # x
            angle_next = -math.atan2(num,den)/math.pi
         
            new_curvature = angle_next-angle_prev
            curvature_chain.append(new_curvature)
        
        convexity = 0
        concavity = 0
        for i in range(len(curvature_chain)):
            if curvature_chain[i] > self.curvature_threshold:
                convexity += 1
            if curvature_chain[i] < -self.curvature_threshold:
                concavity += 1     

        convexity_ratio = convexity / float(i+1)
        concavity_ratio = concavity / float(i+1)

        self.new_feature_values = []
        for ft in features:
            self.new_feature_values.append(eval(ft))
#        [axes_ratio, curvature_ratio, convexity_ratio, area_ratio, vertex_approx, length]
    
    def class_probability(self,obj_type):
        test_point = np.array(self.new_feature_values)
        obj_mu = np.array(obj_type.feature_vec)
        obj_covar = np.matrix(obj_type.feature_covar)
        return norm_pdf_multivariate(test_point,obj_mu,obj_covar)


class colour_class:
        
    def __init__(self,col_name,hsv_min,hsv_max):
        self.name = col_name
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max
        

class feature:
    
    def __init__(self,feat_name):
        self.name = feat_name
        self.mu = 0
        self.var = 0
        self.prev_mu = self.mu
        
    def update_feature(self,new_value,samples):
        self.prev_mu = self.mu
        self.mu = (self.prev_mu * (samples-1) + new_value) / samples
        if samples > 2:
            self.var = self.var * (samples-2) / (samples-1) + (new_value - self.prev_mu)**2 / samples 
        elif samples == 2:
            self.var = ((self.prev_mu - self.mu)**2 + (new_value - self.mu)**2)/2

class object_class:
    
    def __init__(self,obj_type,feature_list):
        self.name = obj_type
        self.samples = 0
        self.features = []
        self.feature_vec = []
        self.feature_var = []
        for ft in range(len(feature_list)):
            self.features.append(feature(feature_list[ft]))
            self.feature_vec.append(self.features[ft].mu)
            self.feature_var.append(self.features[ft].var)
        self.feature_covar = np.diag(np.sqrt(self.feature_var))
            
    
    def update_object(self,new_feature_values,logfile):
        self.samples += 1
        #print self.name, self.samples
        for ft in range(len(new_feature_values)):
            self.features[ft].update_feature(new_feature_values[ft],self.samples)
            self.feature_vec[ft] = self.features[ft].mu
            self.feature_var[ft] = self.features[ft].var
        self.feature_covar = np.diag(np.sqrt(self.feature_var))
        self.save_state(logfile)

    def print_state(self):
        print(self.name, self.samples)
        print(self.feature_vec)
        print(self.feature_covar)        

    def save_state(self,logfile):
        logfile.write(self.name)
        logfile.write(str([self.samples]))
        logfile.write(str(self.feature_vec))
        logfile.write(str(self.feature_var))
        logfile.write('\n')
        

class world:
    
    def __init__(self,folder,world_obj,sequence,features,logfile,colours,hsv_min,hsv_max,classifier):
        
        self.world_objects = world_obj
        n = len(sequence)
        self.known_colours = []
        if colours:
            for co in range(len(colours)):
                self.known_colours.append(colour_class(colours[co],hsv_min[co],hsv_max[co]))
        self.world_experience = []
        self.feature_space = []
        self.labels = []
        for obj in range(len(world_obj)):
            self.world_experience.append(object_class(self.world_objects[obj],features))
            for s in range(n):
                filename = self.world_objects[obj] + str(sequence[s])
                current_image = image(folder,filename,self.known_colours,obj)
                current_contour = contour(current_image.contour,features)  # chain = contours(i)
                self.feature_space.append(current_contour.new_feature_values)
                self.labels.append(obj)
                self.world_experience[obj].update_object(current_contour.new_feature_values,logfile)
        self.classifier = classifier
        self.scaler = StandardScaler()
        self.scaler.fit(self.feature_space)
        X_train = self.scaler.transform(self.feature_space)
        self.classifier.fit(np.asarray(X_train), np.asarray(self.labels))


    def run_pca(self):
        pca = PCA()
        pca.fit(self.feature_space)
        X_transformed = pca.fit_transform(self.feature_space)
        return pca, X_transformed

    def new_classify(self,contour):

        temp = np.asarray(contour.new_feature_values).reshape(1, -1)
        X_test = self.scaler.transform(temp)
        prob = self.classifier.predict_proba(X_test)
        return prob


    def classify(self,contour):
        prob = [];
        for obj in range(len(self.world_experience)):
            prob.append(contour.class_probability(self.world_experience[obj]))
        return prob
        
    def update(self,folder,filename,features,update_world,logfile,colours,obj):
        update_threshold = 0
        current_image = image(folder,filename,self.known_colours,obj)
        current_contour = contour(current_image.contour,features)  # chain = contours(i)
        #prob = self.classify(current_contour)
        prob = self.new_classify(current_contour)
        if update_world and np.max(prob)>update_threshold:
            self.world_experience[np.argmax(prob)].update_object(current_contour.new_feature_values,logfile)
        return prob
