/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iterator>
#include <cmath> 




#include "particle_filter.h"

using namespace std;




static random_device rd; /* the constructor is implemetation-defined */	
// random engine generator
static mt19937 gen(rd()); 

	
	
void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// The number of particles were chosen with an estimated guess
	// a better way would tune the parameter w.r.t error and computational cost
	num_particles = 70;
	
	normal_distribution<double> dx(x, std[0]);
	normal_distribution<double> dy(y, std[1]);
	normal_distribution<double> dtheta(theta, std[2]);

	// init particles
	for (int i = 0; i < num_particles; ++i) 
	{
		Particle p;
		p.id = i;
		p.x = dx(gen);
		p.y = dy(gen);
		p.theta = dtheta(gen);
		p.weight = 1.0;
		particles.push_back(p);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dx_noise(0, std_pos[0]);
	normal_distribution<double> dy_noise(0, std_pos[1]);
	normal_distribution<double> dtheta_noise(0, std_pos[2]);
	
	for (auto& p : particles )
	{
		if (fabs(yaw_rate) < 0.00001) 
		{
			// circular motion with no radial velocity
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else
		{
			// CTRV
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta) );
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t) );
			p.theta += yaw_rate * delta_t;
		}
		
		// add noise 
		p.x += dx_noise(gen);
		p.y += dy_noise(gen);
		p.theta += dtheta_noise(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) 
{
	
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
	// Map predicted obs with observations coming from sensor 
	// the goal is to find the id of the observations which corresponds to predicted
	
	for (LandmarkObs& obs : observations)
	{		
		// find the predicted landmark closest to obs		
		double min_distance = numeric_limits<double>::max();
		unsigned int min_index = 0;
		
		for (size_t i = 0; i < predicted.size(); ++i)
		{			
			double distance = dist (obs.x, obs.y, predicted[i].x, predicted[i].y);
			if (distance < min_distance)
			{
				min_distance = distance;
				min_index = i;
			}		
		}		
		// obs nearest map id
		obs.id = predicted[min_index].id;		
	}							
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs>& observations, const Map &map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	for (auto& p: particles)
	{
		
		// consider landmarks  within range of sensor
		vector<LandmarkObs> l_within_range;
		
		for (auto& landmark : map_landmarks.landmark_list)
		{
			if (fabs(landmark.x_f - p.x) <= sensor_range && fabs(landmark.y_f - p.y) <= sensor_range)
			{
				l_within_range.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
			}
		}
		
		// tranform observations into world coordinates
		// Not working when not using a copy for transformation
		vector<LandmarkObs> observations_in_map;
		for (auto& obs: observations)
		{			
			double x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			double y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

			observations_in_map.push_back(LandmarkObs{ obs.id, x, y });
		}
		
		// use dataAssociation to matchs measurements and landmarks of map 
		dataAssociation(l_within_range, observations_in_map);
		
		// update weight with 2-d gaussian
		
		p.weight = 1.0;
		for (auto& obs: observations_in_map)
		{
			// predicted landmark associated with obs 
			double l_x = 0.0;
			double l_y = 0.0;
			
			size_t index = 0;
			bool found = false ;
			while( !found && index < l_within_range.size())
			{
				if (l_within_range[index].id == obs.id) 
				{
					l_x = l_within_range[index].x;
					l_y = l_within_range[index].y;
					found = true;
				}
				else 
				{
					++index;
				}
			}
			if (!found)
			{
				cout <<"Observation not matching any landmark " << endl;
				return ;
			}
			// calculate weight for this observation with multivariate Gaussian
			double x_error = pow(l_x-obs.x,2)/(2*pow(std_landmark[0], 2));
			double y_error = pow(l_y-obs.y,2)/(2*pow(std_landmark[1], 2));      
      double prob_w = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( x_error + y_error) );
			p.weight *= prob_w;		
		}						
	}
			
}


void ParticleFilter::resample() 
{
	
	
	// 
  vector<double> weights;
  for (auto& p: particles)
	{
    weights.push_back(p.weight);
  }
	
	
	// Implementation of the resampling wheel
	// learnt from AI for robotics course
	
	uniform_int_distribution<int> int_dis(0, num_particles -1);	
	uniform_real_distribution<double> real_dis(0, 1);
	
	int index = int_dis(gen);
	double beta = 0.0; 
	double max_weight = *max_element(weights.begin(), weights.end());
	
	vector<Particle> tmp;
	// start the wheel
	for (int i = 0 ; i < num_particles; ++i)
	{
		beta += real_dis(gen) * 2.0 * max_weight;
		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		tmp.push_back(particles[index]);
	}
	particles = std::move(tmp);
		
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
