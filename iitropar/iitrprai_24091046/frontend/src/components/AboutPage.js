import React from 'react';
import { 
  User, 
  Briefcase, 
  GraduationCap, 
  Github, 
  Linkedin, 
  Mail, 
  Code, 
  Trophy,
  Calendar,
  MapPin,
  ExternalLink,
  Star
} from 'lucide-react';

const AboutPage = () => {
  const skills = [
    'Software Testing', 'Test Automation', 'Selenium WebDriver', 'API Testing',
    'Performance Testing', 'CI/CD', 'Agile/Scrum', 'Java', 'Python', 'JavaScript'
  ];

  const achievements = [
    { icon: Trophy, text: '6+ Years IT Experience' },
    { icon: Code, text: 'SDET Specialist' },
    { icon: Star, text: 'AI & ML Enthusiast' },
    { icon: GraduationCap, text: 'Minor in AI' }
  ];

  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-cyan-500 rounded-3xl p-8 mb-8 text-center">
        <div className="bg-white/20 backdrop-blur-sm w-32 h-32 rounded-full mx-auto mb-6 flex items-center justify-center floating-animation">
          <img 
            src="./img.png" 
            alt="Soham Ghosh" 
            className="w-full h-full object-cover"
          />
        </div>
        <h1 className="text-4xl font-bold text-white mb-2">Soham Ghosh</h1>
        <p className="text-blue-100 text-xl">Software Development Engineer in Test | AI Enthusiast</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Personal Info Card */}
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-6 rounded-3xl shadow-2xl card-hover backdrop-blur-sm border border-white/20">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
              <User className="h-6 w-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white">Personal Information</h2>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-3 text-white/90">
              <MapPin className="h-5 w-5 text-purple-200" />
              <span>Bengaluru, India</span>
            </div>
            <div className="flex items-center space-x-3 text-white/90">
              <Calendar className="h-5 w-5 text-purple-200" />
              <span>6+ Years in IT Industry</span>
            </div>
            <div className="flex items-center space-x-3 text-white/90">
              <Briefcase className="h-5 w-5 text-purple-200" />
              <span>Senior SDET Professional</span>
            </div>
            <div className="flex items-center space-x-3 text-white/90">
              <GraduationCap className="h-5 w-5 text-purple-200" />
              <span>Minor in AI, IIT Ropar</span>
            </div>
          </div>
        </div>

        {/* Professional Experience Card */}
        <div className="bg-gradient-to-br from-cyan-500 to-teal-600 p-6 rounded-3xl shadow-2xl card-hover backdrop-blur-sm border border-white/20">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
              <Briefcase className="h-6 w-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white">Professional Experience</h2>
          </div>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4 border border-white/20">
            <h3 className="text-xl font-bold text-white mb-3">Software Development Engineer in Test (SDET)</h3>
            <p className="text-white/90 leading-relaxed">
              With over 6 years of comprehensive experience in IT, I specialize in software quality assurance, 
              test automation, and ensuring robust software delivery. My expertise spans across various testing 
              methodologies, automation frameworks, and quality engineering practices.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-white/20 rounded-full text-sm text-white">Quality Assurance</span>
              <span className="px-3 py-1 bg-white/20 rounded-full text-sm text-white">Test Automation</span>
              <span className="px-3 py-1 bg-white/20 rounded-full text-sm text-white">Framework Design</span>
            </div>
          </div>
        </div>
      </div>

      {/* Achievements Section */}
      <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-6 rounded-3xl shadow-2xl mb-8 card-hover backdrop-blur-sm border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
            <Trophy className="h-6 w-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white">Key Achievements</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {achievements.map((achievement, index) => (
            <div 
              key={index}
              className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center border border-white/20 hover:bg-white/20 transition-all duration-300"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <achievement.icon className="h-8 w-8 text-white mx-auto mb-2" />
              <p className="text-white font-medium text-sm">{achievement.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Skills Section */}
      <div className="bg-gradient-to-br from-pink-500 to-rose-600 p-6 rounded-3xl shadow-2xl mb-8 card-hover backdrop-blur-sm border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
            <Code className="h-6 w-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white">Technical Skills</h2>
        </div>
        
        <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4 border border-white/20">
          <div className="flex flex-wrap gap-3">
            {skills.map((skill, index) => (
              <span 
                key={index}
                className="px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-white font-medium hover:bg-white/30 transition-all duration-300 hover:scale-105 transform cursor-default"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                {skill}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Contact Section */}
      <div className="bg-gradient-to-br from-violet-500 to-purple-600 p-6 rounded-3xl shadow-2xl card-hover backdrop-blur-sm border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
            <Mail className="h-6 w-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white">Let's Connect</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <a 
            href="https://linkedin.com/in/isohamghosh" 
            target="_blank" 
            rel="noopener noreferrer"
            className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105 transform group"
          >
            <div className="flex items-center space-x-4">
              <div className="bg-blue-500/80 p-3 rounded-lg group-hover:bg-blue-500 transition-colors duration-300">
                <Linkedin className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="text-white font-bold">LinkedIn Profile</h3>
                <p className="text-white/80 text-sm">Connect with me professionally</p>
              </div>
              <ExternalLink className="h-5 w-5 text-white/60 ml-auto group-hover:text-white transition-colors duration-300" />
            </div>
          </a>
          
          <a 
            href="https://github.com/isohamghosh/market-forecast" 
            target="_blank" 
            rel="noopener noreferrer"
            className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105 transform group"
          >
            <div className="flex items-center space-x-4">
              <div className="bg-gray-700/80 p-3 rounded-lg group-hover:bg-gray-700 transition-colors duration-300">
                <Github className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="text-white font-bold">GitHub Profile</h3>
                <p className="text-white/80 text-sm">Check out this project</p>
              </div>
              <ExternalLink className="h-5 w-5 text-white/60 ml-auto group-hover:text-white transition-colors duration-300" />
            </div>
          </a>
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-white/80 text-lg">
            "Passionate about quality engineering and exploring the intersection of AI and software testing."
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;